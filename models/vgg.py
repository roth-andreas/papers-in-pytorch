import torch.nn as nn
from models import building_blocks as bb


class VGG(nn.Module):
    """
    Defines a VGG Network for an input shape of 224x224
    (https://arxiv.org/abs/1409.1556)
    """
    def __init__(self, classes=1000, c_channels=3, dropout=0.5):
        """
        Initializes the VGG Network
        Args:
            classes: Number of output classes
            c_channels: Number of input channels
            dropout: Dropout rate to apply on the fully connected layers
        """
        super(VGG, self).__init__()
        self.convs = nn.Sequential(
            conv_block(1, c_channels, 64),
            # 112 x 112
            conv_block(1, 64, 128, 'maxpool'),
            # 56 x 56
            conv_block(2, 128, 256, 'maxpool'),
            # 28 x 28
            conv_block(2, 256, 512, 'maxpool'),
            # 14 x 14
            conv_block(2, 512, 512, 'maxpool'),
            nn.MaxPool2d(2, 2)
            # 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, classes)
        )

    def forward(self, inputs):
        """
        Defines the forward pass through the Network
        Args:
            inputs: Batched input images of shape (b,channels,224,224)

        Returns: Output of the model with shape (b, classes) as logits

        """
        x = self.convs(inputs)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x


def conv_block(depth, in_dim, out_dim, downsampling=None):
    """
    Defines a conv block for the vgg model
    Args:
        depth: Number of layers for this block
        in_dim: Input channels of this block
        out_dim: Output channels of this block
        downsampling: Downsampling strategy to use

    Returns: Conv block as Module

    """
    return bb.ConvBlock(in_dim,
                        out_dim,
                        downsampling=downsampling,
                        padding=1,
                        batch_norm=False,
                        layer_count=depth)
