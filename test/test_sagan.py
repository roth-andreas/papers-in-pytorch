import unittest
from models import sagan, self_attention_conv
import torch


class GeneratorTest(unittest.TestCase):
    def test_gen_block_shape(self):
        model = sagan.GenBlock(10, 20)
        inputs = torch.ones((4, 10, 16, 16))
        targets = (4, 20, 32, 32)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())

    def test_generator_shape(self):
        model = sagan.Generator(100, 2)
        inputs = torch.randn((4, 100))
        targets = (4, 1, 128, 128)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())
        sac_layers = [m for m in model.modules() if isinstance(m, self_attention_conv.ConvSelfAttention)]
        self.assertEqual(len(sac_layers), 0)

    def test_small_generator_shape(self):
        model = sagan.Generator(100, 2, up_layers=3)
        inputs = torch.randn((4, 100))
        targets = (4, 1, 32, 32)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())
        sac_layers = [m for m in model.modules() if isinstance(m, self_attention_conv.ConvSelfAttention)]
        self.assertEqual(len(sac_layers), 0)

    def test_self_attention_generator(self):
        model = sagan.Generator(100, 2, sa_layer=1)
        inputs = torch.randn((4, 100))
        targets = (4, 1, 128, 128)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())
        sac_layers = [m for m in model.modules() if isinstance(m, self_attention_conv.ConvSelfAttention)]
        self.assertEqual(len(sac_layers), 1)

class DiscriminatorTest(unittest.TestCase):
    def test_disc_block_downsample(self):
        model = sagan.DiscBlock(10, 20)
        inputs = torch.ones((4, 10, 16, 16))
        targets = (4, 20, 8, 8)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())

    def test_disc_block_no_downsample(self):
        model = sagan.DiscBlock(10, 20, downsample=False)
        inputs = torch.ones((4, 10, 16, 16))
        targets = (4, 20, 16, 16)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())

    def test_disc_block_same_channels(self):
        model = sagan.DiscBlock(10, 10, downsample=False)
        inputs = torch.ones((4, 10, 16, 16))
        targets = (4, 10, 16, 16)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())

    def test_disc_block_downsample_last(self):
        model = sagan.DiscBlock(10, 10, downsample=True, downsample_last=True)
        inputs = torch.ones((4, 10, 16, 16))
        targets = (4, 10, 8, 8)
        outputs = model(inputs)
        self.assertEqual(targets, outputs.size())

if __name__ == '__main__':
    unittest.main()
