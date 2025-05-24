import sys
import os
import numpy as np
import tensorflow as tf

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.complex_tensorflow_cnn import ComplexTensorflowCNN
# Assuming you have a data loading function for CNNs, e.g., for MNIST
# from AI_Model_Zoo_Pro.datasets.load_cnn_data import load_cnn_data

def main():
    print("Running Complex TensorFlow CNN Example")

    # --- Data Preparation ---
    # In a real scenario, load your image data here (e.g., MNIST, CIFAR-10)
    # For this example, we'll generate dummy data that mimics image data shape
    input_shape = (28, 28, 1) # Example: MNIST image shape (height, width, channels)
    num_classes = 10 # Example: Number of classes for classification
    num_samples = 100

    # Generate dummy image data (values between 0 and 1)
    x_dummy = np.random.rand(num_samples, *input_shape).astype(np.float32)
    # Generate dummy one-hot encoded labels
    y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes=num_classes)

    print(f"Generated dummy data: x_dummy shape {x_dummy.shape}, y_dummy shape {y_dummy.shape}")

    # --- Model Usage ---
    # Instantiate the model
    cnn_model = ComplexTensorflowCNN(input_shape=input_shape, num_classes=num_classes)

    # Compile the model
    cnn_model.compile()
    print("Model compiled.")

    # Train the model (on dummy data for demonstration)
    print("Training the model...")
    # Note: Training on random data won't yield meaningful results, this is just to show the process
    cnn_model.train(x_dummy, y_dummy, epochs=1, batch_size=32)
    print("Training finished.")

    # Make a prediction on a sample
    sample_input = x_dummy[0:1] # Take the first dummy sample
    prediction = cnn_model.predict(sample_input)
    print(f"Prediction for a sample input: {prediction}")

    # --- Save and Load (Optional) ---
    # save_path = "./complex_cnn_model_saved"
    # cnn_model.save(save_path)
    # print(f"Model saved to {save_path}")

    # loaded_cnn_model = ComplexTensorflowCNN.load(save_path)
    # print(f"Model loaded from {save_path}")

    # Clean up saved model files (TensorFlow saves as a directory)
    # import shutil
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    main()