import torch
import torch.nn as nn

#     Implementation of the original Generative Adversarial Network by Goodfellow et. al.
#     (Paper: https://arxiv.org/abs/1406.2661)


class Generator(nn.Module):
    """
    Implementation of the Generator part
    """
    def __init__(self, z_dim=100, out_dim=(28, 28), hidden_dim=128):
        """
        Initializes the generator
        Args:
            z_dim: Dimension of the latent space
            out_dim: Size of the output
            hidden_dim: Initial size of the hidden dimension
        """
        super(Generator, self).__init__()
        assert len(out_dim) == 2, 'Output shape must have 2 dimensions.'
        self.layers = nn.Sequential(
            *generator_block(z_dim, hidden_dim),
            *generator_block(hidden_dim, hidden_dim * 2),
            *generator_block(hidden_dim * 2, hidden_dim * 4),
            *generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, out_dim[0] * out_dim[1]),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        Defines the forward pass through the generator
        Args:
            inputs: Batched input of random noise with shape (b, z_dim)

        Returns: Batched output of the generator with shape (b, w*h)

        """
        x = self.layers(inputs)
        return x


def generator_block(in_dim, out_dim):
    """
    Defines the layers in a generative block
    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels

    Returns: List of all layers in this block

    """
    return [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(True)
    ]


class Discriminator(nn.Module):
    """
    Implementation of the Discriminator part
    """
    def __init__(self, in_dim=(28, 28), hidden_dim=128):
        """
        Initializes the Discriminator
        Args:
            in_dim: Dimension of the input image as a 2d-tuple
            hidden_dim: Size of the hidden dimension
        """
        super(Discriminator, self).__init__()
        assert len(in_dim) == 2, 'Input shape must have 2 dimensions.'
        flattened_dim = in_dim[0] * in_dim[1]

        self.layers = nn.Sequential(
            *discriminator_block(flattened_dim, hidden_dim * 4),
            *discriminator_block(hidden_dim * 4, hidden_dim * 2),
            *discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Defines the forward pass through the network
        Args:
            inputs: Input images with shape (b, channels, w*h)

        Returns: Singular value for each image as a logit, shape (b, 1)

        """
        x = inputs.view(len(inputs), -1)
        x = self.layers(x)
        return x


def discriminator_block(in_dim, out_dim, pool_size=2):
    """
    Defines the layers in a discriminative block
    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels
        pool_size: Pooling window size to use for Maxout

    Returns: List of all layers in this block

    """
    return [
        nn.Linear(in_dim, out_dim * pool_size),
        Maxout(pool_size),
        nn.Dropout(0.5)
    ]


class Maxout(nn.Module):
    """
    Implementation of the maxout Activation used in the Discriminator
    Paper: https://arxiv.org/pdf/1302.4389.pdf
    """
    def __init__(self, pool_size=3):
        super(Maxout, self).__init__()
        self.pool_size = pool_size

    def forward(self, inputs):
        pool_shape = (*inputs.shape[:1], inputs.shape[1] // self.pool_size, self.pool_size, *inputs.shape[2:])
        max_values, max_indices = inputs.view(pool_shape).max(2)
        return max_values
