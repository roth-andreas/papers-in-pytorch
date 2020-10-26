import unittest
from models import vgg
import torch


class VGGTest(unittest.TestCase):
    def test_output_shape(self):
        model = vgg.VGG(4, 3)
        inputs = torch.ones((4, 3, 224, 224))
        targets = (4, 4)
        outputs = model(inputs)
        self.assertEqual(outputs.size(), targets)


if __name__ == '__main__':
    unittest.main()
