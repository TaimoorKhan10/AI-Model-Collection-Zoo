import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression

class SimpleLinearRegressionModel:
    """A simple linear regression model using scikit-learn."""

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, X, y):
        """Trains the linear regression model."""
        # Scikit-learn models expect X as (n_samples, n_features) and y as (n_samples,)
        if X.ndim == 1:
            X = X.reshape(-1, 1) # Reshape if X is 1D
        if y.ndim > 1:
            y = y.ravel() # Flatten y if it's multi-dimensional

        self.model.fit(X, y)
        self.is_trained = True
        print("Simple Linear Regression Model trained.")

    def predict(self, X):
        """Makes predictions using the trained model."""
        if not self.is_trained:
            print("Warning: Model is not trained. Returning None.")
            return None

        if X.ndim == 1:
            X = X.reshape(-1, 1) # Reshape if X is 1D

        predictions = self.model.predict(X)
        print("Simple Linear Regression Model prediction made.")
        return predictions

    def save(self, file_path):
        """Saves the trained model to a file using pickle."""
        if not self.is_trained:
            print("Warning: Model is not trained. Nothing to save.")
            return

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Simple Linear Regression Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, file_path):
        """Loads a trained model from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                loaded_model = pickle.load(f)
            instance = cls()
            instance.model = loaded_model
            instance.is_trained = True # Assume loaded model is trained
            print(f"Simple Linear Regression Model loaded from {file_path}")
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example Usage (for demonstration/testing)
if __name__ == '__main__':
    # Generate dummy data for linear regression
    X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Input features (must be 2D)
    y_train = np.array([2, 4, 5, 4, 5]) # Target variable

    # Create, train, and predict
    lr_model = SimpleLinearRegressionModel()
    lr_model.train(X_train, y_train)

    X_predict = np.array([6, 7]).reshape(-1, 1)
    predictions = lr_model.predict(X_predict)
    print("Predictions:", predictions)

    # Save and load
    save_file = "./simple_lr_model.pkl"
    lr_model.save(save_file)

    loaded_lr_model = SimpleLinearRegressionModel.load(save_file)
    loaded_predictions = loaded_lr_model.predict(X_predict)
    print("Loaded Predictions:", loaded_predictions)

    # Clean up
    if os.path.exists(save_file):
        os.remove(save_file)
        print(f"Cleaned up {save_file}")