import unittest
from models import self_attention, self_attention_multihead, self_attention_conv
import torch


class SelfAttentionTest(unittest.TestCase):
    def test_something(self):
        d = 20
        d_k = 30
        model = self_attention.SelfAttention(d, d_k)
        inputs = torch.ones((4, 10, d))
        outputs = model(inputs)
        target_size = (4, 10, d_k)
        self.assertEqual(target_size, outputs.shape)

    def test_1d(self):
        d = 1
        d_k = 1
        model = self_attention.SelfAttention(d, d_k)
        inputs = torch.ones((4, d))
        outputs = model(inputs)
        target_size = (4, d, d_k)
        self.assertEqual(target_size, outputs.shape)


class MultiheadTest(unittest.TestCase):
    def test_something(self):
        d = 20
        model = self_attention_multihead.MultiheadSelfAttention(d, 4)
        inputs = torch.ones((4, 10, d))
        outputs = model(inputs)
        target_size = (4, 10, d)
        self.assertEqual(target_size, outputs.shape)

class ConvSelfAttentionTest(unittest.TestCase):
    def test_basic_shape(self):
        d = 20
        d_k = 30
        model = self_attention_conv.ConvSelfAttention(d, d_k)
        inputs = torch.ones((4, d, 28, 28))
        outputs = model(inputs)
        target_size = (4, d, 28, 28)
        self.assertEqual(target_size, outputs.shape)

    def test_shape(self):
        d = 1
        d_k = 30
        model = self_attention_conv.ConvSelfAttention(d, d_k)
        inputs = torch.ones((4, d, 28, 28))
        outputs = model(inputs)
        target_size = (4, d, 28, 28)
        self.assertEqual(target_size, outputs.shape)


if __name__ == '__main__':
    unittest.main()
