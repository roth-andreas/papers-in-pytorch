import torch
import torch.nn as nn
from models import self_attention


class MultiheadSelfAttention(nn.Module):
    """
    Implementation of Multihead Attention as defined in the Paper 'Attention is all you need'
    (https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, d, heads=8):
        """
        Initializes a multi head attention block
        Args:
            d: Input dimension of the data
            heads: Number of heads to use
        """
        super(MultiheadSelfAttention, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(heads):
            self.heads.append(self_attention.SelfAttention(d, d // heads, d // heads))

        self.output_weights = nn.Linear(d, d)

    def forward(self, inputs) -> None:
        """
        Applies the multi head attention block
        Args:
            inputs: Batch of inputs with shape (b, w, d)

        Returns: Batch with applied self attention of shape (b, w, d)

        """
        head_outputs = torch.cat([head(inputs) for head in self.heads], dim=-1)
        return self.output_weights(head_outputs)
