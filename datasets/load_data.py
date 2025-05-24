import pandas as pd

def load_sample_data():
    """Loads a sample dataset for demonstration."""
    # Replace with actual data loading logic
    data = {'feature1': [0.1, 0.2, 0.3], 'feature2': [0.4, 0.5, 0.6], 'target': [0, 1, 0]}
    df = pd.DataFrame(data)
    return df

# Add more data loading functions as needed