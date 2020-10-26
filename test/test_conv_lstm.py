import unittest
from models.conv_lstm import ConvLSTM
import torch


class MyTestCase(unittest.TestCase):
    def test_many_to_one(self):
        feature_size = 10
        hidden_size = 50
        img_size = 28
        model = ConvLSTM(feature_size, hidden_size)
        inputs = torch.zeros((16, 5, feature_size, img_size, img_size))
        hidden_state, cell_state = model(inputs)
        target_size = (16, hidden_size, img_size, img_size)
        self.assertEqual(hidden_state.size(), target_size)

    def test_many_to_many(self):
        feature_size = 10
        hidden_size = 50
        img_size = 28
        model = ConvLSTM(feature_size, hidden_size, return_sequences=True)
        inputs = torch.zeros((16, 5, feature_size, img_size, img_size))
        hidden_seq, (hidden_state, cell_state) = model(inputs)

if __name__ == '__main__':
    unittest.main()
