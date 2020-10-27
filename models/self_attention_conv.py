import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvSelfAttention(nn.Module):
    """
    Implementation of a convolutional self attention block
    as described in 'Self-Attention Generative Adversarial Networks'
    (https://arxiv.org/abs/1805.08318)
    """
    def __init__(self, in_channels, downsample=False):
        """
        Initializes the Self Attention Block
        Args:
            in_channels: Dimension of the Input
        """
        super(ConvSelfAttention, self).__init__()
        h_channels = in_channels // 8

        self.attention_weights = nn.Conv2d(in_channels, 2 * h_channels + in_channels // 2, 1, 1, 0, bias=False)
        self.output_weights = nn.Conv2d(in_channels // 2, in_channels, 1, 1, 0, bias=False)
        self.h_channels = h_channels
        self.downsample = downsample
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Applies self attention to the input

        Args:
            inputs: Input data to self attend

        Returns: Self attended data with output shape (b, in_channels, w, h)

        """
        bs, c, width, height = inputs.size()
        weights = self.attention_weights(inputs)
        query = weights[:, :self.h_channels].view(bs, self.h_channels, width*height)
        key = weights[:, self.h_channels:self.h_channels * 2]
        value = weights[:, 2 * self.h_channels:]

        if self.downsample:
            key = F.max_pool2d(key, 2, 2)
            value = F.max_pool2d(value, 2, 2)

        key = key.view(bs, self.h_channels, -1)
        value = value.view(bs, c // 2, -1)

        s = torch.bmm(query.transpose(1, 2), key)
        logsum = torch.logsumexp(s, dim=2, keepdim=True)
        attn = torch.exp(s - logsum)
        y = torch.bmm(value, attn.transpose(1, 2))
        y = y.view(bs, c // 2, width, height)
        o = self.output_weights(y) * self.gamma + inputs
        return o
