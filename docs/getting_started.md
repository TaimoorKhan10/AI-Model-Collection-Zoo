# Getting Started

Welcome to the AI Model Zoo Pro project! This guide will help you set up the project and run the initial examples.

## Prerequisites

- Python 3.7+
- Git
- Docker (optional, for containerization)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd AI-Model-Zoo-Pro
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source .venv/bin/activate
      ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running Examples

Navigate to the project root directory and run the example script:

```bash
python examples/run_example.py
```

This script demonstrates loading dummy data, normalizing it, and running it through the simple PyTorch model.

You can also run the example for the complex TensorFlow CNN model:

```bash
python examples/run_complex_cnn_example.py
```

This script demonstrates creating, compiling, and training the complex CNN model on dummy data.

You can also run the example for the simple Keras model:

```bash
python examples/run_simple_keras_example.py
```

This script demonstrates creating, compiling, training, saving, and loading the simple Keras model on dummy data.

You can also run the example for the simple RNN model:

```bash
python examples/run_simple_rnn_example.py
```

This script demonstrates creating, compiling, training, saving, and loading the simple RNN model on dummy sequence data.

You can also run the example for the simple Linear Regression model:

```bash
python examples/run_simple_linear_regression_example.py
```

This script demonstrates creating, training, saving, and loading the simple Linear Regression model on dummy data.

You can also run the example for the simple K-Means model:

```bash
python examples/run_simple_kmeans_example.py
```

This script demonstrates creating, training, saving, and loading the simple K-Means model on dummy data.

## Running Tests

Navigate to the project root directory and run the test script:

```bash
python scripts/run_tests.py
```

This will discover and run the unit tests in the `tests` directory.

## Running the API

Navigate to the project root directory and run the API application:

```bash
python -m api.app
```

The API will run on `http://localhost:5000`. You can test the `/predict` endpoint using a tool like `curl` or Postman.

Example `curl` request (replace `[features]` with a list of numbers, e.g., `[0.1, 0.2, ..., 1.0]`):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_pytorch", "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' http://localhost:5000/predict
```

To use the `simple_sklearn` model, change the `model_name` in the request body:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_sklearn", "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' http://localhost:5000/predict
```

To use the `simple_tensorflow` model, change the `model_name` in the request body:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_tensorflow", "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' http://localhost:5000/predict
```

To use the `simple_keras` model, change the `model_name` in the request body:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_keras", "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' http://localhost:5000/predict
```

To use the `complex_tensorflow_cnn` model, change the `model_name` and provide features as a flattened list corresponding to the input shape (e.g., 784 features for a 28x28x1 image):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "complex_tensorflow_cnn", "features": [/* 784 numbers representing a flattened 28x28x1 image */]}' http://localhost:5000/predict
```

To use the `simple_rnn` model, change the `model_name` and provide features as a flattened list corresponding to the input shape (e.g., 5 features for a sequence of length 5 with 1 feature per timestep):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_rnn", "features": [0.1, 0.2, 0.3, 0.4, 0.5]}' http://localhost:5000/predict
```

To use the `simple_linear_regression` model, change the `model_name` and provide features as a list (e.g., `[1.0]` for a single feature):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_linear_regression", "features": [1.0]}' http://localhost:5000/predict
```

To use the `simple_kmeans` model, change the `model_name` and provide features as a list (e.g., `[1.0, 2.0]` for two features):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "simple_kmeans", "features": [1.0, 2.0]}' http://localhost:5000/predict
```

## Running the Webapp

Navigate to the project root directory and run the web application:

```bash
python -m webapp.app
```

The webapp will run on `http://localhost:5001`. Open this URL in your browser to see the index page.