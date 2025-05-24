import unittest
import numpy as np
import os

# Add the project root to the Python path to allow importing modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AI_Model_Zoo_Pro.models.simple_kmeans_model import SimpleKMeansModel

class TestSimpleKMeansModel(unittest.TestCase):

    def setUp(self):
        """Set up dummy data and model for testing."""
        np.random.seed(42)
        # Create data with clear clusters for testing K-Means
        self.X_train = np.vstack([
            np.random.rand(30, 2) * 2 + [1, 1], # Cluster 1
            np.random.rand(30, 2) * 2 + [8, 8], # Cluster 2
            np.random.rand(30, 2) * 2 + [4, 4]  # Cluster 3
        ])
        self.X_test = np.array([[1.5, 1.5], [8.5, 8.5], [4.5, 4.5], [0, 0]])
        self.n_clusters = 3
        self.model = SimpleKMeansModel(n_clusters=self.n_clusters)
        self.save_path = 'test_kmeans_model.pkl'

    def tearDown(self):
        """Clean up any created files."""
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_instantiation(self):
        """Test model instantiation."""
        self.assertIsInstance(self.model, SimpleKMeansModel)
        self.assertEqual(self.model.n_clusters, self.n_clusters)
        self.assertIsNone(self.model.model) # Model should not be trained yet

    def test_train(self):
        """Test model training."""
        self.model.train(self.X_train)
        self.assertIsNotNone(self.model.model) # Model should be trained
        # Check if the underlying KMeans model is fitted
        self.assertTrue(hasattr(self.model.model, 'cluster_centers_'))
        self.assertEqual(self.model.model.n_clusters, self.n_clusters)

    def test_predict(self):
        """Test model prediction."""
        self.model.train(self.X_train)
        predictions = self.model.predict(self.X_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])
        # Basic check: predictions should be within the range of cluster indices
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < self.n_clusters))

    def test_save_load(self):
        """Test model saving and loading."""
        self.model.train(self.X_train)
        original_predictions = self.model.predict(self.X_test)

        self.model.save(self.save_path)
        self.assertTrue(os.path.exists(self.save_path))

        loaded_model = SimpleKMeansModel()
        loaded_model.load(self.save_path)

        self.assertIsNotNone(loaded_model.model)
        loaded_predictions = loaded_model.predict(self.X_test)

        self.assertTrue(np.array_equal(original_predictions, loaded_predictions))

if __name__ == '__main__':
    unittest.main()