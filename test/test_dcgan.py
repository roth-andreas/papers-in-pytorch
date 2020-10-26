import unittest
import torch
from models import dcgan

class GeneratorTest(unittest.TestCase):
    def setUp(self):
        self.model = dcgan.Generator(z_dim=100)

    def test_shape(self):
        self.model = dcgan.Generator(z_dim=100)
        inputs = torch.ones((4, 100))
        outputs = self.model(inputs)
        self.assertEqual(outputs.size(), (4, 1, 64, 64))
        self.assertTrue((outputs <= 1).byte().all())
        self.assertTrue((-1 <= outputs).byte().all())

    def test_bilinear(self):
        self.model = dcgan.Generator(z_dim=100, upsampling='bilinear')
        inputs = torch.ones((4, 100))
        outputs = self.model(inputs)
        self.assertEqual(outputs.size(), (4, 1, 64, 64))
        self.assertTrue((outputs <= 1).byte().all())
        self.assertTrue((-1 <= outputs).byte().all())

class DiscriminatorTest(unittest.TestCase):
    def setUp(self):
        self.model = dcgan.Discriminator()

    def test_shape(self):
        inputs = torch.ones((4, 1, 64, 64))
        outputs = self.model(inputs)
        self.assertEqual(outputs.size(), (4, 1))


if __name__ == '__main__':
    unittest.main()
