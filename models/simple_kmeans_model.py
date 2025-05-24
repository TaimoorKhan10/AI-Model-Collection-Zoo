import numpy as np
import os
import pickle
from sklearn.cluster import KMeans

class SimpleKMeansModel:
    """A simple K-Means clustering model using scikit-learn."""

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10) # n_init explicitly set for KMeans > 0.24
        self.is_trained = False

    def train(self, X):
        """Trains the K-Means clustering model."""
        # Scikit-learn models expect X as (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1) # Reshape if X is 1D

        self.model.fit(X)
        self.is_trained = True
        print("Simple K-Means Model trained.")

    def predict(self, X):
        """Predicts cluster labels for new data."""
        if not self.is_trained:
            print("Warning: Model is not trained. Returning None.")
            return None

        if X.ndim == 1:
            X = X.reshape(-1, 1) # Reshape if X is 1D

        predictions = self.model.predict(X)
        print("Simple K-Means Model prediction made.")
        return predictions

    def save(self, file_path):
        """Saves the trained model to a file using pickle."""
        if not self.is_trained:
            print("Warning: Model is not trained. Nothing to save.")
            return

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Simple K-Means Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, file_path):
        """Loads a trained model from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                loaded_model = pickle.load(f)
            # Instantiate with dummy parameters, actual model is loaded
            instance = cls(n_clusters=loaded_model.n_clusters)
            instance.model = loaded_model
            instance.is_trained = True # Assume loaded model is trained
            print(f"Simple K-Means Model loaded from {file_path}")
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example Usage (for demonstration/testing)
if __name__ == '__main__':
    # Generate dummy data for clustering
    X_train = np.random.rand(100, 2) # 100 samples, 2 features

    # Create, train, and predict
    kmeans_model = SimpleKMeansModel(n_clusters=3)
    kmeans_model.train(X_train)

    X_predict = np.random.rand(10, 2)
    predictions = kmeans_model.predict(X_predict)
    print("Predictions:", predictions)

    # Save and load
    save_file = "./simple_kmeans_model_example.pkl"
    kmeans_model.save(save_file)

    loaded_kmeans_model = SimpleKMeansModel.load(save_file)
    loaded_predictions = loaded_kmeans_model.predict(X_predict)
    print("Loaded Predictions:", loaded_predictions)

    # Clean up
    if os.path.exists(save_file):
        os.remove(save_file)
        print(f"Cleaned up {save_file}")