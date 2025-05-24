import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class SimpleSklearnModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

if __name__ == '__main__':
    # Example usage
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleSklearnModel()
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    print("Sample predictions:", predictions[:5])

    # Save and load example
    model_path = 'simple_sklearn_model.pkl'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    loaded_model = SimpleSklearnModel()
    loaded_model.load(model_path)
    print(f"Model loaded from {model_path}")

    # Clean up dummy file
    import os
    os.remove(model_path)