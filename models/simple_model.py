import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

if __name__ == '__main__':
    # Example usage
    input_dim = 10
    output_dim = 2
    model = SimpleModel(input_dim, output_dim)
    print(f"Model architecture:\n{model}")

    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")