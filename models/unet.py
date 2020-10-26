import torch
import torch.nn as nn
import torch.nn.functional as F

from models import building_blocks, utils


class UNet(nn.Module):
    """
    Implementation of the U-Net Model (https://arxiv.org/abs/1505.04597)
    """

    def __init__(self, filter_sizes=None,
                 skip_connections=True,
                 padding=0,
                 c_channels=1,
                 classes=1,
                 upsampling=None,
                 batch_norm=False,
                 has_residuals=False):
        """ Initializes the UNet Model

        Parameters
        ----------
        filter_sizes : Int Array, optional
            Depth of the filters for each block in the contracting and expanding path
        skip_connections : Boolean, optional
            Whether or not to use skip connections from the contracting to the expanding path
        padding: Int, optional
            Amount of padding to use for convolutions in pixels on each side
        c_channels: Int, optional
            Depth of color channels
        batch_norm: Boolean, optional
            Whether to use batch normalization or not
        """
        super(UNet, self).__init__()
        if filter_sizes is None:
            filter_sizes = [64, 128, 256, 512, 1024]
        self.contractors = nn.ModuleList([ContractingBlock(c_channels,
                                                           filter_sizes[0],
                                                           downsampling=None,
                                                           padding=padding,
                                                           batch_norm=batch_norm,
                                                           has_residuals=has_residuals)])
        self.expanders = nn.ModuleList()
        for index in range(len(filter_sizes) - 1):
            self.contractors.append(ContractingBlock(filter_sizes[index],
                                                     filter_sizes[index + 1],
                                                     padding=padding,
                                                     batch_norm=batch_norm,
                                                     has_residuals=has_residuals))
            self.expanders.append(ExpansiveBlock(filter_sizes[-(index + 1)],
                                                 filter_sizes[-(index + 2)],
                                                 skip_connections,
                                                 padding=padding,
                                                 upsampling=upsampling,
                                                 batch_norm=batch_norm,
                                                 has_residuals=has_residuals))
        self.output = nn.Conv2d(filter_sizes[0], classes, 1, 1, 0)
        utils.init_weights(self.modules())

    def forward(self, inputs):
        outputs = [self.contractors[0](inputs)]
        for layer in self.contractors[1:]:
            # print(outputs[-1].size())
            outputs += [layer(outputs[-1])]
        for index, layer in enumerate(self.expanders):
            # print(outputs[-1].size())
            outputs += [layer(outputs[-1], outputs[len(self.expanders) - index - 1])]

        return self.output(outputs[-1])


class ExpansiveBlock(nn.Module):
    """ Defines one expansive Block of the U-Net

    Parameters
        ----------
        in_dim : Int
            Depth Dimension of the input features
        out_dim: Int
            Depth Dimension of the Output features
        skip_connections : Boolean, optional
            Whether or not to use skip connections from the contracting to the expanding path
        padding: Int, optional
            Amount of padding to use for convolutions in pixels on each side
        upsampling: String, optional
            Which upsampling strategy to use, if None use Deconvolution
        batch_norm: Boolean, optional
            Whether to use batch normalization or not
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 skip_connections,
                 padding=0,
                 upsampling=None,
                 batch_norm=False,
                 has_residuals=False):
        super(ExpansiveBlock, self).__init__()
        self.skip_connections = skip_connections
        self.upsampling = upsampling

        if upsampling is None:
            self.upsample = nn.ConvTranspose2d(in_dim, out_dim, 2, 2, 0)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling, align_corners=True),
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_dim, out_dim, 2, 1, 0)
            )

        self.block = building_blocks.ConvBlock(in_dim if skip_connections else out_dim,
                                               out_dim,
                                               padding=padding,
                                               batch_norm=batch_norm,
                                               has_residual=has_residuals)

        utils.init_weights(self.modules())

    def forward(self, inputs, skip_connected):
        x = self.upsample(inputs)
        if self.skip_connections:
            skip_connected = utils.crop_center(skip_connected, x.size())
            x = torch.cat((x, skip_connected), dim=1)

        return self.block(x)


class ContractingBlock(nn.Module):
    """
    Defines a contracting block for the UNet
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 downsampling='maxpool',
                 padding=0,
                 batch_norm=False,
                 has_residuals=False):
        """
        Initializes the contracting block
        Args:
            in_dim: Number of input channels
            out_dim: Number of output channels
            downsampling: Downsampling strategy to use
            padding: Conv Padding to use
            batch_norm: Whether to use batch norm
            has_residuals: Whether the block has a residual connection
        """
        super(ContractingBlock, self).__init__()
        self.block = building_blocks.ConvBlock(in_dim,
                                               out_dim,
                                               downsampling=downsampling,
                                               padding=padding,
                                               batch_norm=batch_norm,
                                               has_residual=has_residuals)

    def forward(self, inputs):
        return self.block(inputs)
