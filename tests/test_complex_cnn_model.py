import unittest
import sys
import os
import numpy as np
import tensorflow as tf

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.complex_tensorflow_cnn import ComplexTensorflowCNN

class TestComplexTensorflowCNN(unittest.TestCase):

    def setUp(self):
        """Set up a dummy model and data for testing."""
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.model = ComplexTensorflowCNN(input_shape=self.input_shape, num_classes=self.num_classes)
        self.model.compile() # Compile the model for training/prediction

        # Create dummy data
        self.num_samples = 5
        self.x_dummy = np.random.rand(self.num_samples, *self.input_shape).astype(np.float32)
        self.y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, self.num_classes, self.num_samples), num_classes=self.num_classes)

    def test_model_instantiation(self):
        """Test if the model can be instantiated."""
        self.assertIsInstance(self.model, ComplexTensorflowCNN)
        self.assertIsNotNone(self.model.model) # Check if the Keras Sequential model is created

    def test_model_compile(self):
        """Test if the model compiles without errors."""
        # Compilation is done in setUp, this test just checks if setUp ran without error
        self.assertTrue(True) # If we reached here, compile was successful

    def test_model_train(self):
        """Test if the model can run a training step."""
        try:
            # Train for a small number of epochs on dummy data
            self.model.train(self.x_dummy, self.y_dummy, epochs=1, batch_size=2)
            train_successful = True
        except Exception as e:
            print(f"Training failed: {e}")
            train_successful = False
        self.assertTrue(train_successful)

    def test_model_predict(self):
        """Test if the model can make predictions and output shape is correct."""
        predictions = self.model.predict(self.x_dummy)
        self.assertIsNotNone(predictions)
        # Check if the output shape is (num_samples, num_classes)
        self.assertEqual(predictions.shape, (self.num_samples, self.num_classes))

    # Add tests for saving and loading if needed, but requires file system interaction
    # def test_model_save_and_load(self):
    #     """Test saving and loading the model."""
    #     save_path = "./test_complex_cnn_model_saved"
    #     self.model.save(save_path)
    #     self.assertTrue(os.path.exists(save_path))
    #
    #     loaded_model = ComplexTensorflowCNN.load(save_path)
    #     self.assertIsInstance(loaded_model, ComplexTensorflowCNN)
    #     self.assertIsNotNone(loaded_model.model)
    #
    #     # Clean up
    #     import shutil
    #     if os.path.exists(save_path):
    #         shutil.rmtree(save_path)

if __name__ == '__main__':
    unittest.main()