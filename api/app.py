from flask import Flask, request, jsonify
import torch
import sys
import os
import pickle
import tensorflow as tf
import numpy as np

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_model import SimpleModel
from AI_Model_Zoo_Pro.models.simple_sklearn_model import SimpleSklearnModel
from AI_Model_Zoo_Pro.models.simple_tensorflow_model import SimpleTensorflowModel
from AI_Model_Zoo_Pro.models.complex_tensorflow_cnn import ComplexTensorflowCNN
from AI_Model_Zoo_Pro.models.simple_keras_model import SimpleKerasModel
from AI_Model_Zoo_Pro.models.simple_rnn_model import SimpleRNNModel
from AI_Model_Zoo_Pro.models.simple_linear_regression_model import SimpleLinearRegressionModel
from AI_Model_Zoo_Pro.models.simple_kmeans_model import SimpleKMeansModel
# from AI_Model_Zoo_Pro.datasets.load_data import load_dummy_data # Not needed for API inference
from AI_Model_Zoo_Pro.utils.helpers import normalize_data

app = Flask(__name__)

# Dictionary to hold loaded models
# In a real application, you'd manage model loading more robustly (e.g., lazy loading, caching)
loaded_models = {}

def load_model(model_name):
    """Loads a model by name."""
    if model_name in loaded_models:
        return loaded_models[model_name]

    model = None
    model_path = None

    if model_name == 'simple_pytorch':
        # Assuming a saved PyTorch model state_dict or full model
        # For this example, we'll just instantiate the class
        input_dim = 10 # Example dimension
        output_dim = 2 # Example dimension
        model = SimpleModel(input_dim, output_dim)
        # In a real case: model.load_state_dict(torch.load('path/to/pytorch_model.pth'))
        print(f"Loaded PyTorch model: {model_name}")

    elif model_name == 'simple_sklearn':
        # Assuming a saved scikit-learn model pickle file
        # For this example, we'll just instantiate the class
        model = SimpleSklearnModel()
        # In a real case: model.load('path/to/sklearn_model.pkl')
        print(f"Loaded scikit-learn model: {model_name}")

    elif model_name == 'simple_tensorflow':
        # Assuming a saved TensorFlow model directory
        # For this example, we'll just instantiate the class
        input_dim = 10 # Example dimension
        output_dim = 2 # Example dimension
        model = SimpleTensorflowModel(input_dim, output_dim)
        # In a real case: model.load('path/to/tensorflow_model_dir')
        print(f"Loaded TensorFlow model: {model_name}")

    elif model_name == 'complex_tensorflow_cnn':
        # Assuming a saved TensorFlow model directory
        # For this example, we'll just instantiate the class
        # In a real case: model = ComplexTensorflowCNN.load('path/to/cnn_model_dir')
        # For demonstration, we'll create a dummy model instance
        input_shape = (28, 28, 1) # Example shape for CNN
        num_classes = 10 # Example number of classes
        model = ComplexTensorflowCNN(input_shape=input_shape, num_classes=num_classes)
        # Note: A real CNN would likely need weights loaded from a file
        print(f"Loaded TensorFlow CNN model: {model_name}")

    elif model_name == 'simple_keras':
        # Assuming a saved Keras model directory
        # For this example, we'll just instantiate the class
        input_shape = (10,) # Example shape
        model = SimpleKerasModel(input_shape=input_shape)
        # In a real case: model = SimpleKerasModel.load('path/to/keras_model_dir')
        print(f"Loaded Keras model: {model_name}")

    elif model_name == 'simple_rnn':
        # Assuming a saved Keras RNN model directory
        # For this example, we'll just instantiate the class
        # Note: Input shape for RNN is (timesteps, features)
        input_shape = (5, 1) # Example shape
        model = SimpleRNNModel(input_shape=input_shape)
        # In a real case: model = SimpleRNNModel.load('path/to/rnn_model_dir')
        print(f"Loaded Keras RNN model: {model_name}")

    elif model_name == 'simple_linear_regression':
        # Assuming a saved scikit-learn model pickle file
        # For this example, we'll just instantiate the class
        model = SimpleLinearRegressionModel()
        # In a real case: model = SimpleLinearRegressionModel.load('path/to/lr_model.pkl')
        print(f"Loaded scikit-learn Linear Regression model: {model_name}")

    elif model_name == 'simple_kmeans':
        # Assuming a saved scikit-learn model pickle file
        # For this example, we'll just instantiate the class
        model = SimpleKMeansModel()
        # In a real case: model = SimpleKMeansModel.load('path/to/kmeans_model.pkl')
        print(f"Loaded scikit-learn K-Means model: {model_name}")

    # Add more model loading logic here for other models

    if model is not None:
        loaded_models[model_name] = model
        return model
    else:
        return None

# Initial dummy model loading (can be removed if using dynamic loading only)
# input_dim = 10 # This should match the model's expected input
# output_dim = 2 # This should match the model's expected output
# dummy_model = SimpleModel(input_dim, output_dim)
# print("Dummy model loaded for API.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'model_name' not in data:
             return jsonify({'error': 'Missing "model_name" in request body'}), 400
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body'}), 400

        model_name = data['model_name']
        features = data['features']

        model = load_model(model_name)
        if model is None:
            return jsonify({'error': f'Model "{model_name}" not found or failed to load'}), 404

        if not isinstance(features, list) or not all(isinstance(f, (int, float)) for f in features):
             return jsonify({'error': '"features" must be a list of numbers'}), 400

        # Validate input feature dimension based on model type
        expected_input_dim = None
        if hasattr(model, 'input_dim'): # For SimpleModel, SimpleTensorflowModel
            expected_input_dim = model.input_dim
        elif hasattr(model, 'model') and hasattr(model.model, 'input_shape'): # For Keras models
            # Keras input_shape includes batch size as None at index 0
            keras_input_shape = model.model.input_shape
            if len(keras_input_shape) > 1: # Check if input_shape is defined beyond batch size
                 # For simple dense layers, the expected input is flattened
                 # For CNN/RNN, specific checks are handled below
                 if model_name in ['simple_keras', 'simple_tensorflow'] and len(keras_input_shape) == 2: # (None, input_dim)
                     expected_input_dim = keras_input_shape[1]
                 # CNN and RNN have specific reshaping and validation logic below

        if expected_input_dim is not None and len(features) != expected_input_dim:
             return jsonify({'error': f'Invalid number of features for model "{model_name}". Expected {expected_input_dim}, got {len(features)}'}), 400

        # Convert features to appropriate format for the model
        # This is a simplified example; real-world would need more robust handling
        if model_name == 'simple_pytorch':
            input_data = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                output = model(input_data)
                # Assuming classification logits, convert to probabilities
                prediction_result = torch.softmax(output, dim=1).tolist()[0]

        elif model_name == 'simple_sklearn':
            input_data = np.array([features])
            prediction_result = model.predict(input_data).tolist()[0]
            # If it's a classifier, you might want predict_proba
            # prediction_result = model.predict_proba(input_data).tolist()[0]

        elif model_name == 'simple_kmeans':
            # K-Means predict returns the cluster label for each sample
            input_data = np.array([features])
            prediction_result = model.predict(input_data).tolist()[0]

        elif model_name == 'simple_tensorflow':
            input_data = np.array([features], dtype=np.float32)
            prediction_result = model.predict(input_data).tolist()[0]
            # TensorFlow predict returns probabilities directly for softmax output

        elif model_name == 'complex_tensorflow_cnn':
            # Assuming features is a flattened list, reshape for CNN input (batch_size, height, width, channels)
            # This reshaping logic needs to match the expected input_shape of the CNN
            # For a (28, 28, 1) input shape, features should be 28*28 = 784 elements
            try:
                # Example reshape for (28, 28, 1) input
                expected_elements = np.prod(model.model.input_shape[1:])
                if len(features) != expected_elements:
                     return jsonify({'error': f'Invalid number of features for CNN. Expected {expected_elements}, got {len(features)}'}), 400
                input_data = np.array(features, dtype=np.float32).reshape(1, *model.model.input_shape[1:])
                prediction_result = model.predict(input_data).tolist()[0]
            except Exception as reshape_error:
                 return jsonify({'error': f'Error processing features for CNN: {reshape_error}'}), 400

        elif model_name == 'simple_keras':
            # Keras models expect numpy arrays
            input_data = np.array([features], dtype=np.float32)
            prediction_result = model.predict(input_data).tolist()[0]

        elif model_name == 'simple_rnn':
            # RNN models expect numpy arrays with shape (batch_size, timesteps, features)
            # Assuming features is a flattened list of shape (timesteps * features)
            # Reshape features to (1, timesteps, features)
            try:
                # Example reshape for (5, 1) input shape
                expected_elements = np.prod(model.model.input_shape[1:])
                if len(features) != expected_elements:
                     return jsonify({'error': f'Invalid number of features for RNN. Expected {expected_elements}, got {len(features)}'}), 400
                # Need to know the original timesteps and features from the model's input shape
                # Assuming input_shape is (timesteps, features)
                timesteps, features_dim = model.model.input_shape[1:]
                input_data = np.array(features, dtype=np.float32).reshape(1, timesteps, features_dim)
                prediction_result = model.predict(input_data).tolist()[0]
            except Exception as reshape_error:
                 return jsonify({'error': f'Error processing features for RNN: {reshape_error}'}), 400

        elif model_name == 'simple_linear_regression':
            # Scikit-learn models expect numpy arrays with shape (n_samples, n_features)
            # If features is a list [f1, f2, ...], convert to [[f1, f2, ...]]
            input_data = np.array([features])
            prediction_result = model.predict(input_data).tolist()[0]

        else:
             return jsonify({'error': f'Prediction logic not implemented for model type "{model_name}"'}), 501

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        # Log the error in a real application
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

@app.route('/')
def index():
    return "AI Model Zoo Pro API is running!"

if __name__ == '__main__':
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI
    # For development, you can run with debug=True
    # To run: navigate to the project root in terminal and run `python -m api.app`
    # Or from the api directory: `python app.py`
    # Ensure requirements are installed: `pip install -r requirements.txt`
    app.run(debug=True, host='0.0.0.0', port=5000)