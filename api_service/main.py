from fastapi import FastAPI, HTTPException
import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Game Similarity API")

# Initialize model and data
try:
    with open('model.pkl', 'rb') as f:
        state = pickle.load(f)
    
    model = state['model']
    X = state['feature_matrix']
    train_indices = state['train_indices']
    related_slots = state['related_slots']
    train_idx_map = state['train_idx_map']
    slot_id_to_idx = state['slot_id_to_idx']
    
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def get_top_related_slots(df: pd.DataFrame, indices: np.ndarray, train_indices: np.ndarray, full_related_slots: np.ndarray, is_train: bool = True, n_slots: int = 8) -> List[int]:
    """
    Get top related slots from multiple neighbors.
    
    Args:
        df: DataFrame containing the samples to predict for
        indices: Neighbor indices from KNN model
        train_indices: Boolean array indicating training samples
        full_related_slots: Array of related slots for all samples
        is_train: Whether predicting for training samples (True) or test samples (False)
        n_slots: Number of slots to predict
    """
    # Get the mapping from training indices to full dataset indices
    train_to_full = np.where(train_indices)[0]
    
    # Get related slots from all neighbors
    neighbor_slots = []
    # indices[0] is already an array of neighbor indices
    for neighbor_idx in indices[0]:
        # Map the neighbor index from training set to full dataset
        full_idx = train_to_full[neighbor_idx]
        neighbor_slots.extend(full_related_slots[full_idx])
    
    # Count frequencies of slots
    slot_counts = Counter(neighbor_slots)
    
    # Get top n_slots most common slots
    return [slot for slot, _ in slot_counts.most_common(n_slots)]

def predict_for_game(slot_id: int, n_slots: int = 8) -> List[int]:
    """Predict related slots for a single game using its slot_id."""
    try:
        # Convert slot_id to index
        game_idx = slot_id_to_idx[slot_id]
        
        # Create a DataFrame for the single sample
        df_subset = pd.DataFrame(index=[0])
        
        # Get similar games using the training data index
        _, indices = model.kneighbors(
            X=X[game_idx:game_idx+1],
            n_neighbors=model.n_neighbors+1  # +1 because the first result is the game itself
        )
        
        # Get predictions using the same logic as test.py
        return get_top_related_slots(
            df_subset,
            indices,
            train_indices,
            related_slots,
            is_train=False,
            n_slots=n_slots
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Game with slot_id {slot_id} not found")
    except Exception as e:
        logger.error(f"Error predicting for game {slot_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predlagaj/{slot_id}")
async def get_similar_games(slot_id: int, predlogi: int = 8) -> Dict[str, Any]:
    """Get similar games based on slot_id."""
    try:
        # Get predicted related slots
        predicted_slots = predict_for_game(slot_id, n_slots=predlogi)
        
        return {
            "slot_id": slot_id,
            "related_slots": predicted_slots
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 