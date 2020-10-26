import unittest
import models.gan as gan
import torch


class GeneratorTest(unittest.TestCase):
    def setUp(self):
        self.model = gan.Generator(z_dim=100, out_dim=(28, 28))

    def test_shape(self):
        inputs = torch.ones((4, 100))
        outputs = self.model(inputs)
        self.assertEqual(outputs.size(), (4, 28 * 28))
        self.assertTrue((outputs <= 1).byte().all())
        self.assertTrue((0 <= outputs).byte().all())


class DiscriminatorTest(unittest.TestCase):
    def setUp(self):
        self.model = gan.Discriminator()

    def test_shape(self):
        inputs = torch.ones((4, 28*28))
        outputs = self.model(inputs)
        self.assertEqual(outputs.size(), (4, 1))


class MaxoutTest(unittest.TestCase):
    def test_results(self):
        activation = gan.Maxout(2)
        inputs = torch.Tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        targets = torch.Tensor([
            [2, 4],
            [6, 8]
        ])
        outputs = activation(inputs)
        self.assertEqual(outputs.size(), targets.size())
        self.assertTrue(torch.equal(outputs, targets))


if __name__ == '__main__':
    unittest.main()
