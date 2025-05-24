import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os

class ComplexTensorflowCNN:
    """A simple Convolutional Neural Network using TensorFlow."""
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=5, batch_size=32):
        # Assuming y_train is one-hot encoded
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, file_path):
        self.model.save(file_path)

    @classmethod
    def load(cls, file_path):
        loaded_model = tf.keras.models.load_model(file_path)
        instance = cls() # Create an instance to hold the loaded model
        instance.model = loaded_model
        return instance

# Example Usage (requires dummy data generation or actual data)
if __name__ == "__main__":
    # Dummy data (e.g., MNIST-like)
    input_shape = (28, 28, 1)
    num_classes = 10
    x_dummy = np.random.rand(100, *input_shape).astype(np.float32)
    y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes=num_classes)

    # Create, compile, and train the model
    cnn_model = ComplexTensorflowCNN(input_shape=input_shape, num_classes=num_classes)
    cnn_model.compile()
    print("Training the CNN model...")
    cnn_model.train(x_dummy, y_dummy, epochs=1)
    print("Training finished.")

    # Make a prediction
    sample_input = np.random.rand(1, *input_shape).astype(np.float32)
    prediction = cnn_model.predict(sample_input)
    print(f"Prediction for a sample input: {prediction}")

    # Save and load the model
    save_path = "./complex_cnn_model"
    cnn_model.save(save_path)
    print(f"Model saved to {save_path}")

    loaded_cnn_model = ComplexTensorflowCNN.load(save_path)
    print(f"Model loaded from {save_path}")

    # Clean up saved model files
    # Note: TensorFlow saves models as a directory
    # import shutil
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print(f"Cleaned up {save_path}")