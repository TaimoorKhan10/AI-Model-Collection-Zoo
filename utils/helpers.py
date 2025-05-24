import numpy as np

def normalize_data(data):
    """Simple data normalization."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1e-8, std)
    return (data - mean) / std

if __name__ == '__main__':
    dummy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    normalized_data = normalize_data(dummy_data)
    print("Original data:\n", dummy_data)
    print("\nNormalized data:\n", normalized_data)