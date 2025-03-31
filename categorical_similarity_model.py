import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import logging
from scipy.sparse import hstack, csr_matrix
from typing import Dict, List, Tuple
from collections import Counter
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the data."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Remove games with missing related slots
        df = df.dropna(subset=['related_slot_ids'])
        df['related_slot_ids'] = df['related_slot_ids'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df = df[df['related_slot_ids'].apply(len) > 0]  # Remove games with empty related slots
        logger.info(f"After removing games with missing/empty related slots: {df.shape}")
        
        # Convert string representations of lists to actual lists for categorical columns
        categorical_columns = ['FEATURES', 'THEME', 'GENRE', 'OTHER_TAGS', 'TECHNOLOGY', 'OBJECTS', 'TYPE']
        
        def safe_eval(x):
            if not isinstance(x, str):
                return x
            try:
                return eval(x)
            except (SyntaxError, ValueError):
                return [x]
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].apply(safe_eval)
        
        # Process LAYOUT as numeric features
        if 'LAYOUT' in df.columns:
            def parse_layout(x):
                if not isinstance(x, str):
                    return pd.Series([0, 0])
                try:
                    rows, cols = map(int, x.split('x'))
                    return pd.Series([rows, cols])
                except (ValueError, AttributeError):
                    return pd.Series([0, 0])
            
            df[['layout_rows', 'layout_cols']] = df['LAYOUT'].apply(parse_layout)
            df = df.drop('LAYOUT', axis=1)
        
        # Keep numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        excluded_columns = ['top_provider_slots_count', 'related_slots_count']
        numeric_columns = [col for col in numeric_columns if col not in excluded_columns and col != 'slot_id']
        
        # Handle slot_id column
        if 'slot_id' in df.columns:
            # Keep only the first slot_id column if there are duplicates
            slot_id_cols = df.columns[df.columns == 'slot_id'].tolist()
            if len(slot_id_cols) > 1:
                logger.info(f"Found {len(slot_id_cols)} slot_id columns, keeping only the first one")
                # Keep only the first slot_id column
                df = df.rename(columns={slot_id_cols[0]: 'slot_id_temp'})
                for col in slot_id_cols:
                    df = df.drop(col, axis=1)
                df = df.rename(columns={'slot_id_temp': 'slot_id'})
            
            # Convert to integer type
            df['slot_id'] = df['slot_id'].astype(int)
            logger.info(f"slot_id column type after conversion: {df['slot_id'].dtype}")
            logger.info(f"First few slot_ids: {df['slot_id'].head()}")
        
        # Combine numeric and categorical columns
        selected_columns = numeric_columns + categorical_columns + ['related_slot_ids', 'slot_id']
        df = df[selected_columns].copy()  # Make a copy to ensure we have a clean DataFrame
        
        # Verify we only have one slot_id column
        slot_id_cols = df.columns[df.columns == 'slot_id'].tolist()
        if len(slot_id_cols) != 1:
            raise ValueError(f"Expected 1 slot_id column but found {len(slot_id_cols)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features(df: pd.DataFrame, train_indices: np.ndarray = None) -> np.ndarray:
    """Prepare feature matrix with optional train/test split."""
    # Handle numeric features
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != 'related_slot_ids']
    
    # Scale numeric features
    scaler = StandardScaler()
    X_numeric = df[numeric_columns].fillna(df[numeric_columns].median())
    
    if train_indices is not None:
        # Fit scaler on training data only
        X_numeric_scaled = np.zeros_like(X_numeric)
        X_numeric_scaled[train_indices] = scaler.fit_transform(X_numeric.iloc[train_indices])
        X_numeric_scaled[~train_indices] = scaler.transform(X_numeric.iloc[~train_indices])
    else:
        X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    # Process categorical features
    feature_names = list(numeric_columns)
    sparse_matrices = [csr_matrix(X_numeric_scaled)]
    
    # Define categorical columns and their prefixes
    categorical_columns = {
        'FEATURES': 'feature',
        'THEME': 'theme',
        'GENRE': 'genre',
        'OTHER_TAGS': 'tag',
        'TECHNOLOGY': 'tech',
        'OBJECTS': 'object',
        'TYPE': 'type'
    }
    
    # Process each categorical column
    for col, prefix in categorical_columns.items():
        if col in df.columns:
            # Transform the column's values with prefix
            transformed = df[col].apply(lambda x: [f"{prefix}_{item}" for item in x])
            
            # Get MLB for this column
            mlb = MultiLabelBinarizer()
            if train_indices is not None:
                # Fit MLB on training data only
                train_transformed = mlb.fit_transform(transformed.iloc[train_indices])
                test_transformed = mlb.transform(transformed.iloc[~train_indices])
                
                # Create sparse matrix for all data
                X_cat = csr_matrix((len(df), train_transformed.shape[1]))
                X_cat[train_indices] = train_transformed
                X_cat[~train_indices] = test_transformed
            else:
                X_cat = mlb.fit_transform(transformed)
            
            # Add feature names
            feature_names.extend([f"{prefix}_{item}" for item in mlb.classes_])
            sparse_matrices.append(X_cat)
    
    # Combine all features
    X = hstack(sparse_matrices).tocsr()
    
    logger.info(f"Total number of features: {len(feature_names)}")
    logger.info(f"Feature matrix shape: {X.shape}")
    
    return X

def get_top_related_slots(df: pd.DataFrame, indices: np.ndarray, distances: np.ndarray, train_indices: np.ndarray, full_related_slots: np.ndarray, is_train: bool = True, n_slots: int = 8) -> List[Tuple[int, List[int], float]]:
    """
    Get top related slots from multiple neighbors.
    
    Args:
        df: DataFrame containing the samples to predict for
        indices: Neighbor indices from KNN model
        distances: Distances to neighbors
        train_indices: Boolean array indicating training samples
        full_related_slots: Array of related slots for all samples
        is_train: Whether predicting for training samples (True) or test samples (False)
        n_slots: Number of slots to predict
    """
    results = []
    total_games = len(df)
    
    # Get the mapping from training indices to full dataset indices
    train_to_full = np.where(train_indices)[0]
    
    # Process in batches of 1000
    batch_size = 1000
    for i in range(0, total_games, batch_size):
        batch_end = min(i + batch_size, total_games)
        
        for j in range(i, batch_end):
            # Get related slots from all neighbors
            neighbor_slots = []
            # indices[j] is already an array of neighbor indices
            for neighbor_idx in indices[j]:
                # Map the neighbor index from training set to full dataset
                full_idx = train_to_full[neighbor_idx]
                neighbor_slots.extend(full_related_slots[full_idx])
            
            # Count frequencies of slots
            slot_counts = Counter(neighbor_slots)
            
            # Get top n_slots most common slots
            top_slots = [slot for slot, _ in slot_counts.most_common(n_slots)]
            
            # Calculate average distance to neighbors
            avg_distance = np.mean(distances[j])
            
            # For test samples, j is relative to test set, so we need the actual index
            if not is_train:
                game_idx = np.where(~train_indices)[0][j]
            else:
                game_idx = train_to_full[j]  # Map training index to full dataset index
            
            results.append((game_idx, top_slots, avg_distance))
    
    return results

def evaluate_related_slots_predictions(test_df: pd.DataFrame, indices: np.ndarray, distances: np.ndarray, train_indices: np.ndarray, full_related_slots: np.ndarray, n_slots: int = 8) -> Dict[str, float]:
    """Evaluate predictions based on related slot IDs."""
    correct_predictions = 0
    total_games = len(test_df)
    
    # Get top predicted slots for each game
    predictions = get_top_related_slots(test_df, indices, distances, train_indices, full_related_slots, is_train=False, n_slots=n_slots)
    
    # Evaluate each prediction
    for i, (game_idx, predicted_slots, avg_distance) in enumerate(predictions):
        actual_slots = set(test_df['related_slot_ids'].iloc[i])
        predicted_slots = set(predicted_slots)
        
        # Calculate overlap between actual and predicted slots
        overlap = len(actual_slots.intersection(predicted_slots))
        total_unique = len(actual_slots.union(predicted_slots))
        
        # Consider it correct if there's significant overlap
        if overlap > 0 and overlap / total_unique >= 0.5:  # At least 50% overlap
            correct_predictions += 1
    
    accuracy = correct_predictions / total_games
    precision = correct_predictions / total_games
    recall = correct_predictions / total_games
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'correct_predictions': correct_predictions,
        'total_games': total_games
    }

def save_model(filepath: str, model, X: np.ndarray, train_indices: np.ndarray, df: pd.DataFrame) -> None:
    """
    Save the model and all necessary data for predictions.
    
    Args:
        filepath: Path to save the model
        model: Trained KNN model
        X: Feature matrix
        train_indices: Boolean array indicating training samples
        df: Original dataframe with related slots
    """
    try:
        # Save indices mappings for both train and test sets
        train_idx_map = np.where(train_indices)[0]
        test_idx_map = np.where(~train_indices)[0]
        
        # Get the slot_ids from the DataFrame and ensure they are single values
        slot_ids = df['slot_id'].values
        
        # Create mappings using the slot_ids
        slot_id_to_idx = {int(slot_id): idx for idx, slot_id in enumerate(slot_ids)}
        idx_to_slot_id = {idx: int(slot_id) for slot_id, idx in slot_id_to_idx.items()}
        
        state = {
            'model': model,
            'feature_matrix': X,
            'train_indices': train_indices,
            'related_slots': df['related_slot_ids'].values,
            'train_idx_map': train_idx_map,
            'test_idx_map': test_idx_map,
            'slot_id_to_idx': slot_id_to_idx,
            'idx_to_slot_id': idx_to_slot_id,
            'n_neighbors': model.n_neighbors,
            'metric': model.metric,
            'algorithm': model.algorithm
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model and data saved to {filepath}")
        logger.info(f"Number of slot IDs: {len(slot_id_to_idx)}")
        logger.info(f"Example slot_id mapping: {list(slot_id_to_idx.items())[:5]}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_saved_model(filepath: str) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Load a saved model and its associated data.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Tuple containing:
        - Trained KNN model
        - Feature matrix
        - Training indices
        - Related slots array
        - Training indices mapping
        - Test indices mapping
        - Slot ID to index mapping
        - Index to slot ID mapping
    """
    try:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        model = state['model']
        X = state['feature_matrix']
        train_indices = state['train_indices']
        related_slots = state['related_slots']
        train_idx_map = state['train_idx_map']
        test_idx_map = state['test_idx_map']
        slot_id_to_idx = state['slot_id_to_idx']
        idx_to_slot_id = state['idx_to_slot_id']
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model parameters: n_neighbors={model.n_neighbors}, metric={model.metric}")
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Number of training samples: {len(train_idx_map)}")
        logger.info(f"Number of test samples: {len(test_idx_map)}")
        logger.info(f"Number of slot IDs: {len(slot_id_to_idx)}")
        
        return model, X, train_indices, related_slots, train_idx_map, test_idx_map, slot_id_to_idx, idx_to_slot_id
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_for_samples(model: NearestNeighbors, X: np.ndarray, sample_indices: np.ndarray, train_indices: np.ndarray, 
                       related_slots: np.ndarray, is_train: bool = True, n_slots: int = 8) -> List[Tuple[int, List[int], float]]:
    """
    Make predictions for a set of samples.
    
    Args:
        model: Trained KNN model
        X: Feature matrix
        sample_indices: Indices of samples to predict for
        train_indices: Boolean array indicating training samples
        related_slots: Array of related slots
        is_train: Whether predicting for training samples
        n_slots: Number of slots to predict
        
    Returns:
        List of tuples containing (game_idx, predicted_slots, avg_distance)
    """
    # Get predictions for the specified samples only
    distances, indices = model.kneighbors(X[sample_indices])
    
    # Create a DataFrame for the samples with correct size
    df_subset = pd.DataFrame(index=range(np.sum(sample_indices)))
    
    # Get predictions using get_top_related_slots
    predictions = get_top_related_slots(
        df_subset,
        indices,
        distances,
        train_indices,
        related_slots,
        is_train=is_train,
        n_slots=n_slots
    )
    
    return predictions

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Load data
        data_path = 'parsed_data/slot_data.csv'
        df = load_data(data_path)
        
        # Split data into train and test sets
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        train_size = int(0.8 * n_samples)
        train_indices = np.zeros(n_samples, dtype=bool)
        train_indices[indices[:train_size]] = True
        
        logger.info(f"\nData split:")
        logger.info(f"Training set size: {train_size}")
        logger.info(f"Test set size: {n_samples - train_size}")
        
        # Prepare features
        X = prepare_features(df, train_indices)
        
        # Train KNN model with k=3
        logger.info("\nTraining KNN model...")
        model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
        model.fit(X[train_indices])  # Fit only on training data
        
        # Save model and necessary data
        model_path = 'api_service/model.pkl'  # Updated path to match Dockerfile
        save_model(model_path, model, X, train_indices, df)
        
        # Test specific slot IDs
        test_slot_ids = [143, 160, 138]  # The three slot IDs we want to test
        logger.info(f"\nTesting specific slot IDs: {test_slot_ids}")
        
        # Create a DataFrame with just these slots
        test_df = df[df['slot_id'].isin(test_slot_ids)].copy()
        
        # Get indices for these slots in the feature matrix
        test_indices = np.array([df[df['slot_id'] == slot_id].index[0] for slot_id in test_slot_ids])
        
        # Get predictions for these specific slots
        distances, indices = model.kneighbors(X[test_indices])
        
        # Log predictions for each slot
        logger.info("\nPredictions for specific slots:")
        for i, slot_id in enumerate(test_slot_ids):
            predicted_slots = get_top_related_slots(
                test_df.iloc[i:i+1],
                indices[i:i+1],
                distances[i:i+1],
                train_indices,
                df['related_slot_ids'].values,
                is_train=False,
                n_slots=8
            )[0][1]  # Get the predicted slots from the first (and only) result
            
            actual_slots = set(test_df['related_slot_ids'].iloc[i])
            predicted_slots = set(predicted_slots)
            overlap = len(actual_slots.intersection(predicted_slots))
            total_unique = len(actual_slots.union(predicted_slots))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            logger.info(f"\nGame {slot_id}:")
            logger.info(f"Actual slots: {actual_slots}")
            logger.info(f"Predicted slots: {predicted_slots}")
            logger.info(f"Overlap: {overlap} slots")
            logger.info(f"Overlap ratio: {overlap_ratio:.2f}")
            logger.info(f"Average distance to neighbors: {np.mean(distances[i]):.6f}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 