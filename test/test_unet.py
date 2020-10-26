import unittest
from models.unet import UNet
import torch


class DefaultTestCase(unittest.TestCase):
    def setUp(self):
        self.model = UNet(filter_sizes=[2, 4, 8, 16, 32],
                          skip_connections=True,
                          padding=0,
                          c_channels=1,
                          classes=2,
                          upsampling='bilinear',
                          batch_norm=False)

    def test_output_shape(self):
        test_input = torch.randn((4, 1, 572, 572))
        test_output = self.model(test_input)
        self.assertEqual(test_output.size(), (4, 2, 388, 388))

class ResidualTestCase(unittest.TestCase):
    def setUp(self):
        self.model = UNet(filter_sizes=[2, 4, 8, 16, 32],
                          skip_connections=True,
                          padding=0,
                          c_channels=1,
                          classes=2,
                          upsampling='bilinear',
                          batch_norm=False,
                          has_residuals=True)

    def test_output_shape(self):
        test_input = torch.randn((4, 1, 572, 572))
        test_output = self.model(test_input)
        self.assertEqual(test_output.size(), (4, 2, 388, 388))


class AutoencoderTestCase(unittest.TestCase):
    def setUp(self):
        self.model = UNet(filter_sizes=[2, 4, 8, 16, 32],
                          skip_connections=True,
                          padding=1,
                          c_channels=1,
                          classes=2,
                          upsampling='bilinear',
                          batch_norm=False)

    def test_output_shape(self):
        test_input = torch.randn((4, 1, 512, 512))
        test_output = self.model(test_input)
        self.assertEqual(test_output.size(), (4, 2, 512, 512))


if __name__ == '__main__':
    unittest.main()
