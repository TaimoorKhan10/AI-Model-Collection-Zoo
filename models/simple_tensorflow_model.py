import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class SimpleTensorflowModel:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential([
            Dense(10, input_shape=(input_dim,), activation='relu'),
            Dense(output_dim, activation='softmax') # Use softmax for classification
        ])

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

if __name__ == '__main__':
    # Example usage
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    output_dim = 2
    model = SimpleTensorflowModel(input_dim, output_dim)
    model.compile()
    model.train(X_train, y_train, epochs=5)

    predictions = model.predict(X_test)
    print("Sample predictions (probabilities):", predictions[:5])

    # Save and load example
    model_path = 'simple_tensorflow_model'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    loaded_model = SimpleTensorflowModel(input_dim, output_dim) # Need dims for init, but load replaces model
    loaded_model.load(model_path)
    print(f"Model loaded from {model_path}")

    # Clean up dummy file
    import shutil
    shutil.rmtree(model_path)