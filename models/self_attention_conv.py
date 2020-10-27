import torch.nn as nn
import torch


class ConvSelfAttention(nn.Module):
    """
    Implementation of a convolutional self attention block
    as described in 'Self-Attention Generative Adversarial Networks'
    (https://arxiv.org/abs/1805.08318)
    """
    def __init__(self, d, d_k=None):
        """
        Initializes the Self Attention Block
        Args:
            d: Dimension of the Input
            d_k: Key Dimension
        """
        super(ConvSelfAttention, self).__init__()
        if d_k is None:
            d_k = d
        self.attention_weights = nn.Conv2d(d, d_k + d_k + d_k, 1, 1, 0, bias=False)
        self.output_weights = nn.Conv2d(d_k, d, 1, 1, 0, bias=False)
        self.d_k = d_k
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, v):
        """
        Applies self attention to the input

        Args:
            v: Input data to self attend

        Returns: Self attended data with output shape (b, d, d_v)

        """
        bs, c, width, height = v.size()
        weights = self.attention_weights(v)
        query = weights[:, :self.d_k].view(bs, self.d_k, width*height)
        key = weights[:, self.d_k:self.d_k * 2].view(bs, self.d_k, width*height)
        value = weights[:, 2 * self.d_k:].view(bs, self.d_k, width*height)

        s = torch.bmm(query.transpose(1, 2), key)
        logsum = torch.logsumexp(s, dim=2, keepdim=True)
        beta = torch.exp(s - logsum)
        y = torch.bmm(value, beta)
        y = y.view(bs, self.d_k, width, height)
        o = self.output_weights(y) * self.gamma + v
        return o
