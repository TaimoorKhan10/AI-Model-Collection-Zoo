import tensorflow as tf
import numpy as np
import pickle
import os

class SimpleKerasModel:
    """A simple Keras Sequential model for demonstration."""
    def __init__(self, input_shape=(10,)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification output
        ])
        return model

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        # Ensure input data is numpy array and has correct shape
        X_train = np.array(X_train).reshape(-1, *self.input_shape)
        y_train = np.array(y_train)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        print("Training finished.")
        return history

    def predict(self, X_test):
        # Ensure input data is numpy array and has correct shape
        X_test = np.array(X_test).reshape(-1, *self.input_shape)
        predictions = self.model.predict(X_test)
        # For binary classification, return class labels (0 or 1)
        return (predictions > 0.5).astype(int)

    def save(self, file_path):
        # Keras models are typically saved in TensorFlow's SavedModel format or H5 format
        # Using SavedModel format
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        # Load the model using TensorFlow's load_model
        loaded_model = tf.keras.models.load_model(file_path)
        # Create an instance of the class and assign the loaded model
        instance = cls(input_shape=loaded_model.input_shape[1:]) # Infer input shape from loaded model
        instance.model = loaded_model
        print(f"Model loaded from {file_path}")
        return instance

# Example Usage:
if __name__ == "__main__":
    # Generate some dummy data
    X_dummy = np.random.rand(100, 10).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 100).astype(np.float32)

    # Create and compile the model
    keras_model = SimpleKerasModel(input_shape=(10,))
    keras_model.compile()

    # Train the model
    keras_model.train(X_dummy, y_dummy, epochs=5)

    # Make predictions
    predictions = keras_model.predict(X_dummy[:5])
    print("Predictions:", predictions)

    # Save the model
    save_path = "./simple_keras_model_saved"
    keras_model.save(save_path)

    # Load the model
    loaded_keras_model = SimpleKerasModel.load(save_path)

    # Make predictions with the loaded model
    loaded_predictions = loaded_keras_model.predict(X_dummy[:5])
    print("Loaded Predictions:", loaded_predictions)

    # Clean up saved model directory
    # Note: SavedModel format creates a directory, not a single file
    # import shutil
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print(f"Cleaned up {save_path}")