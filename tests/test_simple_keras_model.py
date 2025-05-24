import unittest
import numpy as np
import os
import sys
import shutil

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_keras_model import SimpleKerasModel

class TestSimpleKerasModel(unittest.TestCase):

    def setUp(self):
        """Set up dummy data and model for testing."""
        self.input_shape = (10,)
        self.X_dummy = np.random.rand(10, *self.input_shape).astype(np.float32)
        self.y_dummy = np.random.randint(0, 2, 10).astype(np.float32)
        self.model = SimpleKerasModel(input_shape=self.input_shape)
        self.save_path = "./test_simple_keras_model_saved"

    def tearDown(self):
        """Clean up saved model directory after testing."""
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

    def test_model_instantiation(self):
        """Test if the model is instantiated correctly."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.input_shape[1:], self.input_shape)

    def test_model_compilation(self):
        """Test if the model compiles without errors."""
        try:
            self.model.compile()
            self.assertTrue(True) # Compilation successful
        except Exception as e:
            self.fail(f"Model compilation failed: {e}")

    def test_model_training(self):
        """Test if the model trains without errors."""
        self.model.compile()
        try:
            history = self.model.train(self.X_dummy, self.y_dummy, epochs=1, batch_size=2)
            self.assertIsNotNone(history)
        except Exception as e:
            self.fail(f"Model training failed: {e}")

    def test_model_prediction(self):
        """Test if the model makes predictions without errors."""
        self.model.compile()
        self.model.train(self.X_dummy, self.y_dummy, epochs=1, batch_size=2)
        try:
            predictions = self.model.predict(self.X_dummy[:2])
            self.assertIsNotNone(predictions)
            self.assertEqual(predictions.shape, (2, 1)) # Expecting (batch_size, 1) output for binary classification
        except Exception as e:
            self.fail(f"Model prediction failed: {e}")

    def test_model_save_and_load(self):
        """Test if the model can be saved and loaded."""
        self.model.compile()
        self.model.train(self.X_dummy, self.y_dummy, epochs=1, batch_size=2)

        # Save the model
        self.model.save(self.save_path)
        self.assertTrue(os.path.exists(self.save_path))
        self.assertTrue(os.path.isdir(self.save_path)) # Keras SavedModel is a directory

        # Load the model
        loaded_model_instance = SimpleKerasModel.load(self.save_path)
        self.assertIsNotNone(loaded_model_instance)
        self.assertIsNotNone(loaded_model_instance.model)

        # Test prediction with loaded model
        try:
            loaded_predictions = loaded_model_instance.predict(self.X_dummy[:2])
            self.assertIsNotNone(loaded_predictions)
            self.assertEqual(loaded_predictions.shape, (2, 1))
        except Exception as e:
            self.fail(f"Prediction with loaded model failed: {e}")

if __name__ == '__main__':
    unittest.main()