import numpy as np
import os
import sys

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_linear_regression_model import SimpleLinearRegressionModel

def run_simple_linear_regression_example():
    """Demonstrates the usage of the SimpleLinearRegressionModel."""
    print("Running Simple Linear Regression Model Example...")

    # Generate some dummy data for linear regression
    # X should be 2D (n_samples, n_features)
    X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Input features
    y_train = np.array([2, 4, 5, 4, 5]) # Target variable

    # Create and train the model
    lr_model = SimpleLinearRegressionModel()
    lr_model.train(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    X_predict = np.array([6, 7]).reshape(-1, 1)
    predictions = lr_model.predict(X_predict)
    print("Predictions:", predictions)

    # Save the model
    save_file = "./simple_lr_model_example.pkl"
    lr_model.save(save_file)
    print(f"Model saved to {save_file}")

    # Load the model
    print("Loading the model...")
    loaded_lr_model = SimpleLinearRegressionModel.load(save_file)

    # Make predictions with the loaded model
    print("Making predictions with loaded model...")
    loaded_predictions = loaded_lr_model.predict(X_predict)
    print("Loaded Predictions:", loaded_predictions)

    # Clean up saved model file
    if os.path.exists(save_file):
        os.remove(save_file)
        print(f"Cleaned up {save_file}")

    print("Simple Linear Regression Model Example Finished.")

if __name__ == "__main__":
    run_simple_linear_regression_example()