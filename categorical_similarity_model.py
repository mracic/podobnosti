import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import logging
from collections import Counter
from typing import List, Tuple, Dict, Optional
import pickle
from scipy.sparse import hstack, csr_matrix
import os

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CategoricalGameSimilarityModel:
    def __init__(self, n_neighbors: int = 3, random_state: int = 42):
        """
        Initialize the model.
        
        Args:
            n_neighbors: Number of similar games to consider for prediction
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.model = None
        self.feature_names = None
        self.random_state = random_state
        self.train_indices = None
        self.test_indices = None
        self.game_id_to_idx = None
        self.all_related_slots = None
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        
        # Define categorical columns and their prefixes
        self.categorical_columns = {
            'FEATURES': 'feature',
            'THEME': 'theme',
            'GENRE': 'genre',
            'OTHER_TAGS': 'tag',
            'TECHNOLOGY': 'tech',
            'OBJECTS': 'object',
            'TYPE': 'type'
        }
        
        # Initialize MLB for each categorical column
        self.mlb_transformers = {
            col: MultiLabelBinarizer() 
            for col in self.categorical_columns.keys()
        }

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
            'random_state': self.random_state,
            'game_id_to_idx': self.game_id_to_idx,
            'train_indices': self.train_indices,
            'test_indices': self.test_indices,
            'mlb_transformers': self.mlb_transformers,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'CategoricalGameSimilarityModel':
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
            random_state=state['random_state']
        )
        model.model = state['model']
        model.feature_names = state['feature_names']
        model.all_related_slots = state['all_related_slots']
        model.game_id_to_idx = state['game_id_to_idx']
        model.train_indices = state['train_indices']
        model.test_indices = state['test_indices']
        model.mlb_transformers = state['mlb_transformers']
        model.scaler = state['scaler']
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
            
            # Store all related slots for analysis
            self.all_related_slots = df['related_slot_ids'].values
            
            # Convert string representations of lists to actual lists for categorical columns
            categorical_columns = ['FEATURES', 'THEME', 'GENRE', 'OTHER_TAGS', 'TECHNOLOGY', 'OBJECTS', 'TYPE']
            
            def safe_eval(x):
                if not isinstance(x, str):
                    return x
                try:
                    # Try to evaluate as a Python literal
                    return eval(x)
                except (SyntaxError, ValueError):
                    # If evaluation fails, treat as a single value
                    return [x]
            
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].apply(safe_eval)
            
            # Process LAYOUT as numeric features
            if 'LAYOUT' in df.columns:
                def parse_layout(x):
                    if not isinstance(x, str):
                        return pd.Series([0, 0])  # Default values if not a string
                    try:
                        # Split the layout string (e.g., "5x3") and convert to integers
                        rows, cols = map(int, x.split('x'))
                        return pd.Series([rows, cols])
                    except (ValueError, AttributeError):
                        return pd.Series([0, 0])  # Default values if parsing fails
                
                # Split LAYOUT into rows and columns
                df[['layout_rows', 'layout_cols']] = df['LAYOUT'].apply(parse_layout)
                df = df.drop('LAYOUT', axis=1)  # Remove the original LAYOUT column
            
            # Keep numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            excluded_columns = ['slot_id', 'top_provider_slots_count', 'related_slots_count']
            numeric_columns = [col for col in numeric_columns if col not in excluded_columns]
            
            # Combine numeric and categorical columns
            selected_columns = numeric_columns + categorical_columns + ['related_slot_ids']
            df = df[selected_columns]
            
            # Log missing values
            missing_counts = df[numeric_columns].isna().sum()
            total_rows = len(df)
            logger.info("\nMissing values in numeric columns:")
            for col, count in missing_counts[missing_counts > 0].items():
                percentage = (count / total_rows) * 100
                logger.info(f"{col}: {count} ({percentage:.1f}%)")
            
            return df
            
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
        # First split the data
        np.random.seed(self.random_state)
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        train_size = int(0.8 * n_samples)
        self.train_indices = indices[:train_size]
        self.test_indices = indices[train_size:]
        
        # Split DataFrame
        df_train = df.iloc[self.train_indices].copy()
        df_test = df.iloc[self.test_indices].copy()
        
        # Handle numeric features
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                         if col != 'related_slot_ids']
        
        # Fit scaler on training data only
        X_train_numeric = df_train[numeric_columns].fillna(df_train[numeric_columns].median())
        X_test_numeric = df_test[numeric_columns].fillna(df_test[numeric_columns].median())
        X_train_numeric_scaled = self.scaler.fit_transform(X_train_numeric)
        X_test_numeric_scaled = self.scaler.transform(X_test_numeric)  # Use transform, not fit_transform
        
        # Process categorical features
        feature_names = list(numeric_columns)
        train_sparse_matrices = [csr_matrix(X_train_numeric_scaled)]
        test_sparse_matrices = [csr_matrix(X_test_numeric_scaled)]
        
        # Process each categorical column separately
        for col, prefix in self.categorical_columns.items():
            if col in df.columns:
                # Transform the column's values with prefix
                train_transformed = df_train[col].apply(lambda x: [f"{prefix}_{item}" for item in x])
                test_transformed = df_test[col].apply(lambda x: [f"{prefix}_{item}" for item in x])
                
                # Get MLB for this column
                mlb = self.mlb_transformers[col]
                
                # Fit MLB on training data only
                X_train_cat = mlb.fit_transform(train_transformed)
                X_test_cat = mlb.transform(test_transformed)  # Use transform, not fit_transform
                
                # Add feature names
                feature_names.extend([f"{prefix}_{item}" for item in mlb.classes_])
                train_sparse_matrices.append(X_train_cat)
                test_sparse_matrices.append(X_test_cat)
        
        # Combine all features
        X_train = hstack(train_sparse_matrices).tocsr()
        X_test = hstack(test_sparse_matrices).tocsr()
        self.feature_names = feature_names
        
        logger.info(f"Split data into {len(self.train_indices)} training and {len(self.test_indices)} test samples")
        logger.info(f"Total number of features: {len(feature_names)}")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
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
        
        return predicted_slots, distances[0][1:], dict(slot_counts)  # Skip first distance (self)

    def evaluate_model(self, X_train: np.ndarray, X_test: np.ndarray, n_slots: int = 8) -> Dict[str, float]:
        """
        Evaluate the model using various metrics.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix
            n_slots: Number of slots to predict
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Evaluate on both train and test sets
        train_metrics = self._evaluate_set(X_train, self.train_indices, "train", n_slots)
        test_metrics = self._evaluate_set(X_test, self.test_indices, "test", n_slots)
        
        return {
            'train': train_metrics,
            'test': test_metrics
        }
    
    def _evaluate_set(self, X: np.ndarray, indices: np.ndarray, set_name: str, n_slots: int) -> Dict[str, float]:
        """Helper method to evaluate a specific set."""
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
            predicted, _, _ = self.predict_related_slots(i, n_slots=n_slots)
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
    # Initialize model
    model = CategoricalGameSimilarityModel(n_neighbors=3, random_state=42)
    
    try:
        # Load data
        data_path = 'parsed_data/slot_data.csv'
        df = model.load_data(data_path)
        
        # Prepare features and split data
        X_train, X_test = model.prepare_features(df)
        
        # Train model
        model.train(X_train)
        
        # Test if a game appears as its own nearest neighbor
        test_idx = 0
        distances, indices = model.model.kneighbors(
            X=model.model._fit_X[test_idx:test_idx+1],
            n_neighbors=model.n_neighbors+1
        )
        
        # Check if the game itself is the first neighbor
        threshold = 1e-10
        if indices[0][0] == test_idx and distances[0][0] < threshold:
            logger.info("✓ Confirmed: The game appears as its own nearest neighbor with distance ~0")
        else:
            logger.info("✗ Warning: The game does not appear as its own nearest neighbor")
        
        # Evaluate model on both train and test sets
        metrics = model.evaluate_model(X_train, X_test, n_slots=8)
        
        # Show example predictions
        n_samples = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
        for idx in range(min(3, n_samples)):
            predicted_slots, distances, slot_frequencies = model.predict_related_slots(idx)
            actual_slots = model.all_related_slots[idx]
            
            logger.info(f"\nGame {idx}:")
            logger.info(f"Actual: {actual_slots}")
            logger.info(f"Predicted: {predicted_slots}")
            logger.info("Top 5 slot frequencies:")
            for slot, freq in sorted(slot_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  Slot {slot}: {freq} times")
        
        # Save the trained model
        model.save('api_service/categorical_model.pkl')
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 