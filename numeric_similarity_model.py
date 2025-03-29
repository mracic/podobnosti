import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import logging
from collections import Counter
from typing import List, Tuple, Dict, Optional
import pickle

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SimpleGameSimilarityModel:
    def __init__(self, n_neighbors: int = 3, max_missing_threshold: float = 0.5, random_state: int = 42):
        """
        Initialize the model.
        
        Args:
            n_neighbors: Number of similar games to consider for prediction
            max_missing_threshold: Maximum allowed ratio of missing values in features
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.model = None
        self.feature_names = None
        self.max_missing_threshold = max_missing_threshold
        self.all_related_slots = None
        self.game_id_to_idx = None
        self.random_state = random_state
        self.train_indices = None
        self.test_indices = None
        
    def save(self, filepath: str) -> None:
        """
        Save the model state to a file.
        
        Args:
            filepath: Path to save the model
        """
        state = {
            'model': self.model,
            'feature_names': self.feature_names,
            'all_related_slots': self.all_related_slots,
            'n_neighbors': self.n_neighbors,
            'max_missing_threshold': self.max_missing_threshold,
            'game_id_to_idx': self.game_id_to_idx,
            'random_state': self.random_state,
            'train_indices': self.train_indices,
            'test_indices': self.test_indices
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'SimpleGameSimilarityModel':
        """
        Load a saved model state from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(
            n_neighbors=state['n_neighbors'],
            max_missing_threshold=state['max_missing_threshold'],
            random_state=state['random_state']
        )
        model.model = state['model']
        model.feature_names = state['feature_names']
        model.all_related_slots = state['all_related_slots']
        model.game_id_to_idx = state['game_id_to_idx']
        model.train_indices = state['train_indices']
        model.test_indices = state['test_indices']
        logger.info(f"Model loaded from {filepath}")
        return model

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and prepare the data.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with selected features
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Remove games with missing related slots
            df = df.dropna(subset=['related_slot_ids'])
            df['related_slot_ids'] = df['related_slot_ids'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            df = df[df['related_slot_ids'].apply(len) > 0]  # Remove games with empty related slots
            logger.info(f"After removing games with missing/empty related slots: {df.shape}")
            
            # Store game ID to index mapping
            self.game_id_to_idx = {game_id: idx for idx, game_id in enumerate(df['slot_id'])}
            
            # Keep only numeric columns and related_slot_ids
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            # Exclude identifier and target-related columns
            excluded_columns = ['slot_id', 'top_provider_slots_count', 'related_slots_count']
            numeric_columns = [col for col in numeric_columns if col not in excluded_columns]
            df = df[numeric_columns + ['related_slot_ids']]
            
            # Store all related slots for analysis
            self.all_related_slots = df['related_slot_ids'].values
            
            # Log missing values
            missing_counts = df[numeric_columns].isna().sum()
            total_rows = len(df)
            logger.info("\nMissing values in numeric columns:")
            for col, count in missing_counts[missing_counts > 0].items():
                percentage = (count / total_rows) * 100
                logger.info(f"{col}: {count} ({percentage:.1f}%)")
            
            # Select columns with acceptable missing value ratio
            valid_columns = [col for col in numeric_columns 
                           if missing_counts[col] / total_rows <= self.max_missing_threshold]
            
            # Add related_slot_ids back if it was dropped
            if 'related_slot_ids' not in valid_columns:
                valid_columns.append('related_slot_ids')
            
            # Store feature names
            self.feature_names = [col for col in valid_columns if col != 'related_slot_ids']
            
            logger.info(f"\nSelected {len(self.feature_names)} features with acceptable missing value ratio")
            return df[valid_columns]
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and split into train/test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing:
            - Training feature matrix
            - Test feature matrix
        """
        # Create feature matrix
        X = df[self.feature_names].values
        
        # Handle missing values with median for each column
        X = pd.DataFrame(X, columns=self.feature_names)
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info(f"Imputed {col} with median value: {median_val:.2f}")
        
        # Split the data
        X_train, X_test, self.train_indices, self.test_indices = train_test_split(
            X.values, np.arange(len(X)), 
            test_size=0.3, 
            random_state=self.random_state
        )
        
        logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples")
        return X_train, X_test

    def train(self, X: np.ndarray) -> None:
        """
        Train the KNN model.
        
        Args:
            X: Training feature matrix
        """
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric='cosine',
            algorithm='brute'
        )
        self.model.fit(X)
        logger.info("Model training completed")

    def predict_related_slots(self, 
                            game_idx: int, 
                            n_slots: Optional[int] = None,
                            n_neighbors: Optional[int] = None) -> Tuple[List[int], List[float], Dict[int, int]]:
        """
        Predict related slots for a given game index.
        
        Args:
            game_idx: Index of the game to predict for
            n_slots: Number of slots to predict (default: same as actual related slots)
            n_neighbors: Number of similar games to consider (default: self.n_neighbors)
            
        Returns:
            Tuple containing:
            - List of predicted slot IDs
            - List of distances to similar games
            - Dictionary of slot frequencies
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        # Get similar games
        distances, indices = self.model.kneighbors(
            X=self.model._fit_X[game_idx:game_idx+1],
            n_neighbors=n_neighbors+1  # +1 because the first result is the game itself
        )
        
        # Get related slots from similar games
        similar_slots = []
        for idx in indices[0][1:]:  # Skip the game itself
            similar_slots.extend(self.all_related_slots[idx])
        
        # Count slot frequencies
        slot_counts = Counter(similar_slots)
        
        # If n_slots not specified, use the same number as actual related slots
        if n_slots is None:
            n_slots = len(self.all_related_slots[game_idx])
        
        # Get the most common slots
        predicted_slots = [slot for slot, _ in slot_counts.most_common(n_slots)]
        
        return predicted_slots, distances[0][1:], dict(slot_counts)

    def evaluate_model(self, X: np.ndarray, indices: np.ndarray, set_name: str = "set", n_slots: int = 8) -> Dict[str, float]:
        """
        Evaluate the model using various metrics.
        
        Args:
            X: Feature matrix
            indices: Indices of the games being evaluated
            set_name: Name of the set being evaluated (e.g., "train" or "test")
            n_slots: Number of slots to predict
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Calculate metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i, idx in enumerate(indices):
            # Get actual related slots
            actual = set(self.all_related_slots[idx])
            if not actual:  # Skip if no related slots
                continue
                
            # Get predicted related slots
            predicted, _, _ = self.predict_related_slots(i, n_slots=n_slots)  # Use i instead of idx for feature matrix
            predicted = set(predicted)
            
            # Calculate metrics
            true_positives = len(actual.intersection(predicted))
            precision = true_positives / len(predicted) if predicted else 0
            recall = true_positives / len(actual) if actual else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Calculate average metrics
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        
        # Log results
        logger.info(f"\n{set_name.capitalize()} Set Evaluation Results:")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Recall: {avg_recall:.4f}")
        logger.info(f"Average F1 Score: {avg_f1:.4f}")
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }

def main():
    # Initialize model with 50% missing value threshold and random seed
    model = SimpleGameSimilarityModel(n_neighbors=3, max_missing_threshold=0.5, random_state=42)
    
    try:
        # Load data
        data_path = 'parsed_data/slot_data.csv'
        df = model.load_data(data_path)
        
        # Prepare features and split data
        X_train, X_test = model.prepare_features(df)
        
        # Train model
        model.train(X_train)
        
        # Evaluate model on both sets
        train_metrics = model.evaluate_model(X_train, model.train_indices, "train")
        test_metrics = model.evaluate_model(X_test, model.test_indices, "test")
        
        # Save the trained model directly to api_service directory
        model.save('api_service/model.pkl')
        
        # Show example prediction with a test set game
        test_idx = model.test_indices[0]  # Use first test set game
        predicted_slots, distances, slot_frequencies = model.predict_related_slots(test_idx)
        
        logger.info("\nExample Prediction (Test Set Game):")
        logger.info(f"Game {test_idx} actual related slots: {model.all_related_slots[test_idx]}")
        logger.info(f"Predicted related slots: {predicted_slots}")
        logger.info("\nSlot frequencies in similar games:")
        for slot, freq in sorted(slot_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"Slot {slot}: appears {freq} times")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 