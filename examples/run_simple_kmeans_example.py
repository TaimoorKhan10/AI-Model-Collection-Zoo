import numpy as np
import os

# Add the project root to the Python path to allow importing modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_kmeans_model import SimpleKMeansModel

def run_simple_kmeans_example():
    """Demonstrates the usage of the SimpleKMeansModel."""
    print("Running Simple K-Means Example...")

    # 1. Generate dummy data
    # For K-Means, let's create some clustered data
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10 # 100 samples, 2 features
    # Add some structure to make clusters visible
    X[:30, :] += 2
    X[30:60, :] += 8
    X[60:, :] += 5

    # 2. Instantiate and train the model
    n_clusters = 3 # Example: look for 3 clusters
    kmeans_model = SimpleKMeansModel(n_clusters=n_clusters)
    print(f"Training K-Means model with {n_clusters} clusters...")
    kmeans_model.train(X)
    print("Training complete.")

    # 3. Make predictions
    # Predict the cluster for a few new data points
    new_data = np.array([[2.5, 2.5], [8.1, 7.9], [5.2, 5.5], [0.1, 0.1]])
    predictions = kmeans_model.predict(new_data)
    print(f"Predictions for new data {new_data.tolist()}: {predictions.tolist()}")

    # 4. Save and load the model
    save_path = 'simple_kmeans_model.pkl'
    print(f"Saving model to {save_path}...")
    kmeans_model.save(save_path)
    print("Model saved.")

    print(f"Loading model from {save_path}...")
    loaded_model = SimpleKMeansModel()
    loaded_model.load(save_path)
    print("Model loaded.")

    # Verify loaded model predictions
    loaded_predictions = loaded_model.predict(new_data)
    print(f"Predictions from loaded model: {loaded_predictions.tolist()}")
    assert np.array_equal(predictions, loaded_predictions), "Loaded model predictions do not match!"
    print("Loaded model predictions match original predictions.")

    # 5. Clean up the saved model file
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}.")

    print("Simple K-Means Example finished.")

if __name__ == "__main__":
    run_simple_kmeans_example()