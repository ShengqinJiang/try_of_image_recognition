import unittest
import torch
import sys
import os

# 获取当前文件的目录，然后获取父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.models.cnn_model import SimpleCNN

class TestCNNModel(unittest.TestCase):
    def test_model_creation(self):
        model = SimpleCNN(num_classes=10)
        self.assertIsNotNone(model)
        
    def test_model_forward(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()