import torch.nn as nn


def crop_center(imgs, target_size):
    """
    Crops the center out of a batch of images with a target size
    Args:
        imgs: Batch of images to crop with shape (b, c, w, h)
        target_size: Size to crop imgs to with shape (b, c, w, h)

    Returns: None

    """
    start_x = (imgs.size(2) - target_size[2]) // 2
    start_y = (imgs.size(3) - target_size[3]) // 2
    return imgs[:, :, start_x:start_x + target_size[2], start_y:start_y + target_size[3]]


def add_spectral_normalization(module):
    """
    Adds spectral normalization to the module if it's a conv layer
    Args:
        module: Module to add spectral normalization to

    Returns: Module with spectral normalization

    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return nn.utils.spectral_norm(module)
    else:
        return module


def init_weights(modules):
    """
    Initializes the weights of the modules following Kaiming He Initialization
    (Source: https://arxiv.org/abs/1502.01852)
    Args:
        modules: Modules to initialize the weights for

    Returns: None

    """
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight.data, 1)
            nn.init.constant_(layer.bias.data, 0)


def init_weights_normal(modules):
    """
    We initialize the weights to the normal distribution
    with mean 0 and standard deviation 0.02
    Args:
        modules: Module to init the weights for

    Returns: None

    """
    for layer in modules:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            nn.init.normal_(layer.weight, 0.0, 0.02)
        if isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            nn.init.constant_(layer.bias, 0)