import unittest
import numpy as np
import os
import sys
import shutil

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_linear_regression_model import SimpleLinearRegressionModel

class TestSimpleLinearRegressionModel(unittest.TestCase):

    def setUp(self):
        """Set up dummy data and model for testing."""
        # Dummy data for linear regression
        self.X_dummy = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Input features (must be 2D)
        self.y_dummy = np.array([2, 4, 5, 4, 5]) # Target variable
        self.model = SimpleLinearRegressionModel()
        self.save_file = "./test_simple_lr_model.pkl"

    def tearDown(self):
        """Clean up saved model file after testing."""
        if os.path.exists(self.save_file):
            os.remove(self.save_file)

    def test_model_instantiation(self):
        """Test if the model is instantiated correctly."""
        self.assertIsNotNone(self.model.model)

    def test_model_training(self):
        """Test if the model trains without errors."""
        try:
            self.model.train(self.X_dummy, self.y_dummy)
            self.assertTrue(self.model.is_trained)
        except Exception as e:
            self.fail(f"Model training failed: {e}")

    def test_model_prediction(self):
        """Test if the model makes predictions without errors after training."""
        self.model.train(self.X_dummy, self.y_dummy)
        X_predict = np.array([6, 7]).reshape(-1, 1)
        try:
            predictions = self.model.predict(X_predict)
            self.assertIsNotNone(predictions)
            self.assertEqual(predictions.shape, (2,))
        except Exception as e:
            self.fail(f"Model prediction failed: {e}")

    def test_model_prediction_before_training(self):
        """Test prediction before training (should return None)."""
        predictions = self.model.predict(self.X_dummy[:2])
        self.assertIsNone(predictions)

    def test_model_save_and_load(self):
        """Test if the model can be saved and loaded."""
        self.model.train(self.X_dummy, self.y_dummy)

        # Save the model
        self.model.save(self.save_file)
        self.assertTrue(os.path.exists(self.save_file))

        # Load the model
        loaded_model_instance = SimpleLinearRegressionModel.load(self.save_file)
        self.assertIsNotNone(loaded_model_instance)
        self.assertTrue(loaded_model_instance.is_trained)

        # Test prediction with loaded model
        X_predict = np.array([6, 7]).reshape(-1, 1)
        try:
            loaded_predictions = loaded_model_instance.predict(X_predict)
            self.assertIsNotNone(loaded_predictions)
            self.assertEqual(loaded_predictions.shape, (2,))
        except Exception as e:
            self.fail(f"Prediction with loaded model failed: {e}")

if __name__ == '__main__':
    unittest.main()