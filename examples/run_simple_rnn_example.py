import numpy as np
import os
import sys
import shutil

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_rnn_model import SimpleRNNModel
# from AI_Model_Zoo_Pro.datasets.load_data import load_dummy_data # Example if data loading was needed
# from AI_Model_Zoo_Pro.utils.helpers import normalize_data # Example if normalization was needed

def run_simple_rnn_example():
    """Demonstrates the usage of the SimpleRNNModel."""
    print("Running Simple RNN Model Example...")

    # Generate some dummy sequence data
    # (num_samples, timesteps, features)
    num_samples = 100
    timesteps = 5
    features = 1
    input_shape = (timesteps, features)
    X_dummy = np.random.rand(num_samples, timesteps, features).astype(np.float32)
    y_dummy = np.random.randint(0, 2, num_samples).astype(np.float32)

    # Create and compile the model
    rnn_model = SimpleRNNModel(input_shape=input_shape)
    rnn_model.compile()

    # Train the model
    print("Training the model...")
    rnn_model.train(X_dummy, y_dummy, epochs=5)

    # Make predictions
    print("Making predictions...")
    predictions = rnn_model.predict(X_dummy[:5])
    print("Predictions:", predictions)

    # Save the model
    save_path = "./simple_rnn_model_saved_example"
    rnn_model.save(save_path)
    print(f"Model saved to {save_path}")

    # Load the model
    print("Loading the model...")
    loaded_rnn_model = SimpleRNNModel.load(save_path)

    # Make predictions with the loaded model
    print("Making predictions with loaded model...")
    loaded_predictions = loaded_rnn_model.predict(X_dummy[:5])
    print("Loaded Predictions:", loaded_predictions)

    # Clean up saved model directory
    # Note: SavedModel format creates a directory, not a single file
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print(f"Cleaned up {save_path}")

    print("Simple RNN Model Example Finished.")

if __name__ == "__main__":
    run_simple_rnn_example()