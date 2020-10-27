import torch.nn as nn
import torch
from models import self_attention_conv as sac, utils


class Generator(nn.Module):
    """
    Generator Network for the Self Attention Generative Adversarial Network
    following the paper (https://arxiv.org/abs/1805.08318)
    """
    def __init__(self, z_dim, filter_dim, c_channels=1, sa_layer=-1, up_layers=5, spectral_norm=True):
        """
        Initializes the Generator
        Args:
            z_dim: Hidden dimension of the input noise
            filter_dim: Dimension of the final filter, filter dims before are (2**i)
            c_channels: Number of output color channels
            sa_layer: Position of the self attention layer, -1 for no self attention
            up_layers: Number of upsampling layers
            spectral_norm: Whether to apply spectral normalization to all conv layers
        """
        super(Generator, self).__init__()
        filter_dims = [2**(up_layers-1)] + [2**i for i in range(up_layers-1, -1, -1)]
        self.from_latent = nn.Linear(z_dim, 4 * 4 * filter_dims[0] * filter_dim)

        layers = []
        for i in range(1, len(filter_dims)):
            layers.append(
                GenBlock(filter_dims[i - 1] * filter_dim, filter_dims[i] * filter_dim))
            if i == sa_layer:
                layers.append(
                    sac.ConvSelfAttention(filter_dims[i] * filter_dim, downsample=True))
        layers += [
            nn.BatchNorm2d(filter_dims[-1] * filter_dim),
            nn.ReLU(True),
            nn.Conv2d(filter_dims[-1] * filter_dim, c_channels, 3, 1, 1),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)

        if spectral_norm:
            self.apply(utils.add_spectral_normalization)

    def forward(self, inputs):
        """
        Defines the forward pass for the Generator
        Args:
            inputs: Noise vector of dimension (b, z_dim)

        Returns:
            Generated image with shape (b, c_channels, 4 * (2 ** up_layers), 4 * (2 ** up_layers))
        """
        x = self.from_latent(inputs)
        x = x.view(len(x), -1, 4, 4)
        x = self.layers(x)
        return x


class GenBlock(nn.Module):
    """
    Creates a generator upsampling block with 2 conv layers
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes a single generative block with 2 conv layers
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(GenBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        self.residual_con = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, inputs):
        """
        Defines the forward pass
        Args:
            inputs: Batched input with shape (b, in_channels, w, h)

        Returns:
            Batched output with shape (b, out_channels, w*2, h*2)
        """
        block_out = self.layers(inputs)
        residual_out = self.residual_con(inputs)
        return block_out + residual_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        dim = 10
        self.layers = nn.Sequential(
            DiscBlock(dim, dim),
            DiscBlock(dim, dim * 2, downsample_last=True, start_activation=True),
            DiscBlock(dim * 2, dim * 4, downsample_last=True, start_activation=True),
            DiscBlock(dim * 4, dim * 8, downsample_last=True, start_activation=True),
            DiscBlock(dim * 8, dim * 16, downsample_last=True, start_activation=True),
            DiscBlock(dim * 16, dim * 16, downsample=False, start_activation=True),
            nn.ReLU(True),
        )
        self.final_layers = nn.Linear(dim * 16, 1)

    def forward(self, inputs):
        x = self.layers(inputs)
        x = torch.sum(x, (2, 3))
        return self.final_layers(x)

class DiscBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample=True,
                 start_activation=False,
                 downsample_last=False):
        super(DiscBlock, self).__init__()

        layers = []
        if start_activation:
            layers.append(nn.ReLU(True))
        layers += [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        ]
        if downsample:
            layers.append(nn.AvgPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

        if downsample or (in_channels != out_channels):
            residual_layers = [nn.Conv2d(in_channels, out_channels, 1, 1, 0)]

            if downsample:
                residual_layers.insert(downsample_last, nn.AvgPool2d(2, 2))
            self.residual = nn.Sequential(*residual_layers)

    def forward(self, inputs):
        block_out = self.layers(inputs)
        if hasattr(self, 'residual'):
            inputs = self.residual(inputs)
        return block_out + inputs
