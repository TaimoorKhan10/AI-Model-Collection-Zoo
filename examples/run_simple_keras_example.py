import numpy as np
import os
import sys

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_keras_model import SimpleKerasModel
# from AI_Model_Zoo_Pro.datasets.load_data import load_dummy_data # Example if data loading was needed
# from AI_Model_Zoo_Pro.utils.helpers import normalize_data # Example if normalization was needed

def run_simple_keras_example():
    """Demonstrates the usage of the SimpleKerasModel."""
    print("Running Simple Keras Model Example...")

    # Generate some dummy data
    # In a real scenario, you would load your dataset here
    input_shape = (10,)
    X_dummy = np.random.rand(100, *input_shape).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 100).astype(np.float32)

    # Create and compile the model
    keras_model = SimpleKerasModel(input_shape=input_shape)
    keras_model.compile()

    # Train the model
    print("Training the model...")
    keras_model.train(X_dummy, y_dummy, epochs=5)

    # Make predictions
    print("Making predictions...")
    predictions = keras_model.predict(X_dummy[:5])
    print("Predictions:", predictions)

    # Save the model
    save_path = "./simple_keras_model_saved_example"
    keras_model.save(save_path)
    print(f"Model saved to {save_path}")

    # Load the model
    print("Loading the model...")
    loaded_keras_model = SimpleKerasModel.load(save_path)

    # Make predictions with the loaded model
    print("Making predictions with loaded model...")
    loaded_predictions = loaded_keras_model.predict(X_dummy[:5])
    print("Loaded Predictions:", loaded_predictions)

    # Clean up saved model directory
    # Note: SavedModel format creates a directory, not a single file
    # import shutil
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print(f"Cleaned up {save_path}")

    print("Simple Keras Model Example Finished.")

if __name__ == "__main__":
    run_simple_keras_example()