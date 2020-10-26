import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    """
    Implements a Self Attention Layer
    """
    def __init__(self, d, d_k=None, d_v=None):
        """
        Initializes the Self Attention Block
        Args:
            d: Dimension of the Input
            d_k: Key Dimension
            d_v: Value (Output) Dimension
        """
        super(SelfAttention, self).__init__()
        if d_k is None:
            d_k = d
        if d_v is None:
            d_v = d_k
        self.attention_weights = nn.Linear(d, d_k + d_k + d_v)
        self.d_k = d_k

    def forward(self, v):
        """
        Applies self attention to the input

        Args:
            v: Input data to self attend

        Returns: Self attended data with output shape (b, d, d_v)

        """
        if len(v.shape) == 2:
            v = v.unsqueeze(dim=-1)
        weights = self.attention_weights(v)
        query = weights[..., :self.d_k]
        key = weights[..., self.d_k:self.d_k * 2]
        value = weights[..., 2 * self.d_k:]

        d = query.shape[-1]
        s = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(d)
        logsum = torch.logsumexp(s, dim=2, keepdim=True)
        w = torch.exp(s - logsum)
        y = torch.bmm(w, value)
        return y
