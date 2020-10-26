import torch.nn as nn

from models import utils


class Generator(nn.Module):
    """
    Generator Network Module following DCGAN guidelines
    """
    def __init__(self, z_dim=100, h_dim=1024, channels=1, upsampling=None, spectral_norm=True):
        """
        Initializes the Generator Network
        Args:
            z_dim: Dimension of the latent space
            h_dim: Dimension of the initial hidden space
            channels: Number of output channels
            upsampling: Upsampling strategy to use
            spectral_norm: Whether to use spectral normalization
        """
        super(Generator, self).__init__()
        self.start = nn.Linear(z_dim, h_dim * 4 * 4)
        self.layers = nn.Sequential(
            gen_block(h_dim, h_dim // 2, upsampling=upsampling),
            gen_block(h_dim // 2, h_dim // 4, upsampling=upsampling),
            gen_block(h_dim // 4, h_dim // 8, upsampling=upsampling),
            gen_block(h_dim // 8, channels, activation=False, upsampling=upsampling),
            nn.Sigmoid()
        )
        utils.init_weights_normal(self.modules())
        if spectral_norm:
            self.apply(utils.add_spectral_normalization)

    def forward(self, inputs):
        """
        Defines the forward pass of the network
        Args:
            inputs: Batch of inputs with z_dim values for each sample

        Returns: Output of the network in shape (b, channels, w, h)

        """
        x = self.start(inputs)
        x = x.view(-1, 1024, 4, 4)
        x = self.layers(x)
        return x


def gen_block(in_dim, out_dim, activation=True, upsampling=None):
    """
    Creates a convolutional generative block
    Args:
        in_dim: Number of Input channels
        out_dim: Number of Output channels
        activation: Whether to use an activation and batch norm
        upsampling: Decides which upsampling strategy to use

    Returns:

    """
    if upsampling is None:
        layers = [
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, padding=1),
        ]
    else:
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        ]
    if activation:
        layers += [
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, True),
        ]
    return nn.Sequential(
        *layers
    )


def disc_block(input_channels, output_channels, kernel_size=4, stride=2, activation=True, spectral_norm=False):
    """
    Creates a discriminator block
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        kernel_size: Kernel size of the convolution
        stride: Stride of the convolution
        activation: Whether to use an activation
        spectral_norm: Whether to use spectral normalization

    Returns: Module with one convolutional layer and optionally batch norm and an activation

    """
    conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size, stride)
    if spectral_norm:
        conv_layer = nn.utils.spectral_norm(conv_layer)
    if activation:
        return nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    else:
        return nn.Sequential(
            conv_layer,
        )


class Discriminator(nn.Module):
    """
    Discriminator Network Module following DCGAN guidelines
    """
    def __init__(self, channels=1, hidden_dim=64, spectral_norm=False):
        """
        Initializes the model
        Args:
            channels: Number of input channels
            hidden_dim: Hidden dimension of channels
            spectral_norm: Whether to apply spectral normalization to all conv layers
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            disc_block(channels, hidden_dim),
            disc_block(hidden_dim, hidden_dim * 2),
            disc_block(hidden_dim * 2, 1, activation=False),
        )
        self.output = nn.Linear(36, 1)

        utils.init_weights_normal(self.modules())
        if spectral_norm:
            self.apply(utils.add_spectral_normalization)

    def forward(self, inputs):
        """
        Defines a batched forward pass through the discriminator
        Args:
            inputs: Inputs of shape (b, channels, w, h)

        Returns: Outputs of the network with shape (b, 1) without an activation on the output

        """
        x = self.disc(inputs)
        x = x.view(len(x), -1)
        return self.output(x)
