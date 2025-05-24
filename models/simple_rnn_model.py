import tensorflow as tf
import numpy as np
import os

class SimpleRNNModel:
    """A simple Keras RNN model for sequence data demonstration."""
    def __init__(self, input_shape=(None, 1), units=32):
        # input_shape: (timesteps, features) - None for variable timesteps
        self.input_shape = input_shape
        self.units = units
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.SimpleRNN(self.units, return_sequences=False), # return_sequences=True if stacking RNN layers
            tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification output
        ])
        return model

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        # Ensure input data is numpy array and has correct shape (batch_size, timesteps, features)
        # Assuming X_train is list of sequences or numpy array (num_samples, timesteps, features)
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        # Reshape y_train if necessary (e.g., from (num_samples,) to (num_samples, 1))
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        print("Training finished.")
        return history

    def predict(self, X_test):
        # Ensure input data is numpy array and has correct shape (batch_size, timesteps, features)
        # Assuming X_test is list of sequences or numpy array (num_samples, timesteps, features)
        X_test = np.array(X_test, dtype=np.float32)
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
        # Infer input shape and units from loaded model
        # Note: Inferring original 'units' might be tricky from loaded model config alone
        # For simplicity, we'll just pass the loaded model directly or infer shape
        # A more robust approach might save config separately or use a custom loading function
        input_shape = loaded_model.input_shape[1:]
        # Attempt to infer units from the first RNN layer if it exists
        units = 32 # Default or try to find in config
        for layer in loaded_model.layers:
            if isinstance(layer, tf.keras.layers.SimpleRNN):
                units = layer.units
                break

        instance = cls(input_shape=input_shape, units=units)
        instance.model = loaded_model
        print(f"Model loaded from {file_path}")
        return instance

# Example Usage:
if __name__ == "__main__":
    # Generate some dummy sequence data
    # (num_samples, timesteps, features)
    num_samples = 100
    timesteps = 5
    features = 1
    X_dummy = np.random.rand(num_samples, timesteps, features).astype(np.float32)
    y_dummy = np.random.randint(0, 2, num_samples).astype(np.float32)

    # Create and compile the model
    rnn_model = SimpleRNNModel(input_shape=(timesteps, features))
    rnn_model.compile()

    # Train the model
    rnn_model.train(X_dummy, y_dummy, epochs=5)

    # Make predictions
    predictions = rnn_model.predict(X_dummy[:5])
    print("Predictions:", predictions)

    # Save the model
    save_path = "./simple_rnn_model_saved"
    rnn_model.save(save_path)

    # Load the model
    loaded_rnn_model = SimpleRNNModel.load(save_path)

    # Make predictions with the loaded model
    loaded_predictions = loaded_rnn_model.predict(X_dummy[:5])
    print("Loaded Predictions:", loaded_predictions)

    # Clean up saved model directory
    # Note: SavedModel format creates a directory, not a single file
    # import shutil
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print(f"Cleaned up {save_path}")