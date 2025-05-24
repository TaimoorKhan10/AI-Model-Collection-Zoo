import unittest
import os

# Discover and run tests
# Assumes tests are in the 'tests' directory and files are named test_*.py

if __name__ == '__main__':
    # Get the absolute path to the tests directory
    tests_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
    
    # Add the project root to the Python path to allow importing modules
    project_root = os.path.join(os.path.dirname(__file__), '..')
    import sys
    sys.path.insert(0, project_root)

    print(f"Discovering tests in: {tests_dir}")
    loader = unittest.TestLoader()
    # The pattern 'test_*.py' will automatically include test_complex_cnn_model.py, test_simple_keras_model.py, test_simple_rnn_model.py, test_simple_linear_regression_model.py, and test_simple_kmeans_model.py
    suite = loader.discover(tests_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")