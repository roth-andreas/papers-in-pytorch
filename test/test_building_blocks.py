import unittest

from torch.nn.modules.module import ModuleAttributeError

from models import building_blocks as bb
import torch


class MyTestCase(unittest.TestCase):
    def test_basic(self):
        model = bb.ConvBlock(32, 32)
        inputs = torch.zeros((2, 32, 24, 24))
        outputs = model(inputs)
        self.assertEqual((2, 32, 20, 20), outputs.shape)

    def test_padding(self):
        model = bb.ConvBlock(32, 32, padding=1)
        inputs = torch.zeros((2, 32, 24, 24))
        outputs = model(inputs)
        self.assertTrue(len(model.net) == 6)
        self.assertEqual((2, 32, 24, 24), outputs.shape)

    def test_residual(self):
        model = bb.ConvBlock(32, 32, padding=1, has_residual=True)
        inputs = torch.zeros((2, 32, 24, 24))
        outputs = model(inputs)
        with self.assertRaises(ModuleAttributeError):
            model.shortcut
        self.assertTrue(len(model.net) == 6)
        self.assertEqual((2, 32, 24, 24), outputs.shape)

    def test_downsampling(self):
        model = bb.ConvBlock(32, 32, padding=1, downsampling='conv', has_residual=True)
        inputs = torch.zeros((2, 32, 24, 24))
        outputs = model(inputs)
        self.assertIsNotNone(model.shortcut)
        self.assertTrue(len(model.net) == 6)
        self.assertTrue(len(model.shortcut) == 3)
        self.assertEqual((2, 32, 12, 12), outputs.shape)

    def test_residual_maxpooling(self):
        model = bb.ConvBlock(32, 32, padding=1, downsampling='maxpool', has_residual=True)
        inputs = torch.zeros((2, 32, 24, 24))
        outputs = model(inputs)
        self.assertIsNotNone(model.shortcut)
        self.assertTrue(len(model.net) == 7)
        self.assertTrue(len(model.shortcut) == 3)
        self.assertEqual((2, 32, 12, 12), outputs.shape)

    def test_batchnorm(self):
        model = bb.ConvBlock(32, 32, batch_norm=False)
        inputs = torch.zeros((2, 32, 24, 24))
        outputs = model(inputs)
        self.assertTrue(len(model.net) == 4)
        self.assertEqual((2, 32, 20, 20), outputs.shape)


if __name__ == '__main__':
    unittest.main()
