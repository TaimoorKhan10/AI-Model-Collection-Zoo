import torch
from AI_Model_Zoo_Pro.models.simple_model import SimpleModel
from AI_Model_Zoo_Pro.datasets.load_data import load_dummy_data
from AI_Model_Zoo_Pro.utils.helpers import normalize_data

# Load dummy data
X, y = load_dummy_data(num_samples=50, num_features=10)

# Normalize features
X_normalized = normalize_data(X.values) # Convert pandas DataFrame to numpy array

# Initialize and use the model
input_dim = X_normalized.shape[1]
output_dim = 2 # Assuming binary classification for this example
model = SimpleModel(input_dim, output_dim)

# Convert numpy array to PyTorch tensor
dummy_input_tensor = torch.tensor(X_normalized, dtype=torch.float32)

# Get model output
output = model(dummy_input_tensor)

print("Example run complete.")
print(f"Input data shape: {X_normalized.shape}")
print(f"Model output shape: {output.shape}")