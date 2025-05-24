import unittest
import torch
from AI_Model_Zoo_Pro.models.simple_model import SimpleModel

class TestSimpleModel(unittest.TestCase):

    def test_model_output_shape(self):
        input_dim = 10
        output_dim = 2
        model = SimpleModel(input_dim, output_dim)
        dummy_input = torch.randn(1, input_dim)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, output_dim))

    def test_model_forward_pass(self):
        input_dim = 5
        output_dim = 3
        model = SimpleModel(input_dim, output_dim)
        dummy_input = torch.randn(1, input_dim)
        # Just check if forward pass runs without error
        try:
            model(dummy_input)
            passed = True
        except Exception:
            passed = False
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()