import torch.nn as nn
import torch
import torch.nn.functional as F

from models import utils


def ConvLayer(in_dim, out_dim, kernel_size=3, stride=2, padding=1, batch_norm=True):
    """
    Defines a simple convolutional layer consisting of a Convolution, optionally Batch Normalization and an activation
    Args:
        in_dim: Input feature channels
        out_dim: Output feature channels
        kernel_size: Kernel size of the convolution
        stride: Stride of the convolution
        padding: Padding of the convolution
        batch_norm: Whether to use Batch Normalization

    Returns: List of the layers

    """
    layers = [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_dim))
    layers.append(nn.ReLU(True))
    return layers


class ConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 downsampling=None,
                 padding=0,
                 batch_norm=True,
                 has_residual=False,
                 bottleneck_dim=None,
                 layer_count=2):
        """
        Defines a convolutional consisting of a 2 convolutional layers
        Args:
            in_dim: Input channels for the block
            out_dim: Output channels of the block
            downsampling: Whether to use downsampling as the first operation and which type of downsampling to use
            padding: Padding for the convolutions
            batch_norm: Whether to use Batch Normalization
            layer_count: Count of convolutional layers in this block
            has_residual: Whether to have a residual connection in this block
        """
        super(ConvBlock, self).__init__()

        # Defines the feedforward layers
        layers = []
        if downsampling == 'maxpool':
            layers.append(nn.MaxPool2d(2, 2))

        if bottleneck_dim is not None:
            layers += [
                *ConvLayer(in_dim, bottleneck_dim, 1, 2 if downsampling else 1, 0, batch_norm=True),
                *ConvLayer(bottleneck_dim, bottleneck_dim, 3, 1, 1, batch_norm=True),
                *ConvLayer(bottleneck_dim, out_dim, 1, 1, 0, batch_norm=True)
            ]
        else:
            layers += ConvLayer(in_dim, out_dim, 3, 2 if downsampling == 'conv' else 1, padding, batch_norm=batch_norm)
            for i in range(1, layer_count):
                layers += ConvLayer(out_dim, out_dim, 3, 1, padding, batch_norm=batch_norm)
        self.net = nn.Sequential(*layers)

        # Defines the residual connection
        if has_residual and (
                (in_dim != out_dim) or downsampling is not None
        ):
            stride = 2 if downsampling is not None else 1
            shortcut_layer = ConvLayer(in_dim, out_dim, 1, stride, 0, batch_norm=batch_norm)
            self.shortcut = nn.Sequential(*shortcut_layer)
        self.has_residual = has_residual

    def forward(self, inputs):
        """
        Defines the forward pass through the Conv Block
        Args:
            inputs: Batched Input with shape (b, channels, w, h)

        Returns: Batched output with shape (b, new_channels, new_w, new_h)

        """
        x = inputs
        fx = self.net(inputs)

        if self.has_residual:
            if hasattr(self, 'shortcut'):
                x = self.shortcut(x)
            if x.shape != fx.shape:
                x = utils.crop_center(x, fx.size())
            fx = F.relu(torch.add(fx, x))
        return fx
