import torch.nn as nn
from models import building_blocks as bb, utils


class ResNet(nn.Module):
    """
    Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    """

    def __init__(self,
                 c_channels=3,
                 classes=1000,
                 block_sizes=[2, 2, 2, 2],
                 feature_sizes=[64, 64, 128, 256, 512],
                 is_bottleneck=False,
                 skip_connections=True,
                 initial_kernel_size=7,
                 initial_stride=2,
                 initial_max_pool=True
                 ):
        """
        Initializes a Deep Residual Network
        Args:
            c_channels: Number of Input color channels
            classes: Number of output classes
            block_sizes: Depth of each block
            feature_sizes: Width of each block
            is_bottleneck: Whether to use bottleneck blocks
            skip_connections: Whether to use skip connections
            initial_kernel_size: Initial conv kernel size
            initial_stride: Initial stride
            initial_max_pool: Whether to use max pooling initially
        """
        assert len(feature_sizes) >= 2, 'feature_sizes needs to have at least 2 values'
        assert len(block_sizes) == len(feature_sizes) - 1, 'block_sizes should have length of feature_sizes -1'
        super(ResNet, self).__init__()
        layers = [*bb.ConvLayer(c_channels,
                                feature_sizes[0],
                                initial_kernel_size,
                                initial_stride,
                                initial_kernel_size // 2,
                                batch_norm=True)]
        if initial_max_pool:
            layers += [nn.MaxPool2d(2, 2)]

        for i in range(len(block_sizes)):
            layers += block(block_sizes[i], feature_sizes[i], feature_sizes[i + 1],
                            downsampling='conv' if i > 0 else None,
                            skip_connections=skip_connections, is_bottleneck=is_bottleneck)

        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_sizes[-1], classes)
        ]
        self.net = nn.Sequential(
            *layers
        )
        utils.init_weights(self.modules())

    def forward(self, inputs):
        """
        Defines the forward pass through the network
        Args:
            inputs: Batched Input with shape (b, channels, w, h)

        Returns: Output of the network as logits with shape (b, classes)

        """
        return self.net(inputs)


def block(block_count=3,
          in_dim=64,
          out_dim=64,
          downsampling=None,
          skip_connections=True,
          is_bottleneck=False):
    """
    Defines a block containing one or multiple conv blocks
    Args:
        block_count: Amount of blocks to create with the same settings
        in_dim: Number of input channels
        out_dim: Number of output channels
        downsampling: Downsampling strategy to use
        skip_connections: Whether to use skip connections
        is_bottleneck: Whether the blocks are bottleneck

    Returns: List of Conv block modules

    """
    layers = []

    for i in range(block_count):
        layers += [bb.ConvBlock(in_dim if i == 0 else out_dim,
                                out_dim,
                                downsampling=downsampling if i == 0 else None,
                                has_residual=skip_connections,
                                padding=1,
                                bottleneck_dim=out_dim // 4 if is_bottleneck else None)]
    return layers
