from fastapi import FastAPI, HTTPException
from numeric_similarity_model import SimpleGameSimilarityModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Game Similarity API")

# Initialize model
try:
    # Load pre-trained model
    model = SimpleGameSimilarityModel.load('model.pkl')
    logger.info("Pre-trained model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.get("/predlagaj/{game_id}")
async def get_similar_games(game_id: int, predlogi: int = 8):
    """
    Get similar games for a given game ID.
    
    Args:
        game_id: ID of the game to find similar games for
        predlogi: Number of similar games to return (default: 8)
        
    Returns:
        List of similar game IDs and their distances
    """
    try:
        # Find the index of the game using the mapping
        if game_id not in model.game_id_to_idx:
            raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
        game_idx = model.game_id_to_idx[game_id]
        
        # Get predictions
        predicted_slots, distances, _ = model.predict_related_slots(game_idx, n_slots=predlogi)
        
        return {
            "game_id": game_id,
            "similar_games": [
                {"slot_id": slot_id, "distance": float(dist)}
                for slot_id, dist in zip(predicted_slots, distances)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 