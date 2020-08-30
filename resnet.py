import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,
                 block_count=3,
                 in_dim=64,
                 out_dim=64,
                 downsampling=False,
                 skip_connections=True,
                 is_bottleneck=False):
        super(Block, self).__init__()
    
        layers = []
        if is_bottleneck:
            layers += [BottleneckBlock(in_dim, out_dim, downsampling, skip_connections=skip_connections, bottleneck_dim=out_dim // 4)]   
        else:
            layers += [BasicBlock(2, in_dim, out_dim, downsampling, skip_connections=skip_connections)]

        for i in range(block_count - 1):
            if is_bottleneck:
                layers += [BottleneckBlock(out_dim, out_dim, False, skip_connections=skip_connections, bottleneck_dim=out_dim//4)]
            else:
                layers += [BasicBlock(2, in_dim, out_dim, False, skip_connections=skip_connections)]
        self.block = nn.Sequential(*layers)
  
    def forward(self, inputs):
        return self.block(inputs)

class BasicBlock(nn.Module):
    def __init__(self,
                 layer_count=2,
                 in_dim=64,
                 out_dim=64,
                 downsampling=False,
                 skip_connections=True):
        super(BasicBlock, self).__init__()

        layers = []
        if downsampling:
            layers += [nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True)]
            layer_count -= 1

            if skip_connections:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1, 2, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(True)
                )

        for i in range(layer_count):
            layers += [nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True)]

        self.residual = nn.Sequential(*layers)
        self.downsampling = downsampling
        self.skip_connections = skip_connections


    def forward(self, inputs):
        x = inputs
        fx = self.residual(inputs)

        if self.skip_connections:
            if self.downsampling:
                x = self.shortcut(x)
            fx = torch.add(fx, x)
    
        return F.relu(fx)

class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_dim=64,
                 out_dim=64,
                 downsampling=False,
                 skip_connections=True,
                 bottleneck_dim=64):
        super(BottleneckBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, bottleneck_dim, 1, 2 if downsampling else 1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(True),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(True),
            nn.Conv2d(bottleneck_dim, out_dim, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

        if skip_connections and in_dim != out_dim:
            self.shortcut = nn.Sequential(
              nn.Conv2d(in_dim, out_dim, 1, 2 if downsampling else 1, bias=False),
              nn.BatchNorm2d(out_dim),
              nn.ReLU(True)
            )
            self.apply_skip_transform = True 
        else:
            self.apply_skip_transform = False

        self.skip_connections = skip_connections


    def forward(self, inputs):
        x = inputs
        fx = self.residual(inputs)
        if self.skip_connections:
            if self.apply_skip_transform:
                x = self.shortcut(x)
            fx = torch.add(fx, x)
    
        return F.relu(fx)


class ResNet(nn.Module): 
    '''
    Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    '''
    def __init__(self,
                 c_channels=3,
                 classes=1000,
                 block_sizes=[2,2,2,2],
                 feature_sizes=[64,64,128,256,512],
                 is_bottleneck=False,
                 skip_connections=True,
                 initial_kernel_size=7,
                 initial_stride=2,
                 initial_max_pool=True
                ):
        assert len(feature_sizes) >= 2, 'feature_sizes needs to have at least 2 values'
        assert len(block_sizes) == len(feature_sizes) - 1, 'block_sizes should have length of feature_sizes -1'
        super(ResNet, self).__init__()
        layers = [nn.Conv2d(in_channels=c_channels,
                      out_channels=feature_sizes[0],
                      kernel_size=initial_kernel_size,
                      stride=initial_stride,
                      padding=initial_kernel_size // 2,
                      bias=False),
                  nn.BatchNorm2d(feature_sizes[0]),
                  nn.ReLU(True)]
        if initial_max_pool:
            layers += [nn.MaxPool2d(2,2)]


        layers += [Block(block_sizes[0], feature_sizes[0], feature_sizes[1], downsampling=False, skip_connections=skip_connections, is_bottleneck=is_bottleneck)]
        for i in range(1,len(block_sizes)):
            layers += [Block(block_sizes[i], feature_sizes[i], feature_sizes[i+1], downsampling=True, skip_connections=skip_connections, is_bottleneck=is_bottleneck)]
        body = nn.Sequential(*layers)

        top = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_sizes[-1], classes)
          )
        self.net = nn.Sequential(
            body,
            top
        )
        self.init_weights()

    def forward(self, inputs):
        return self.net(inputs)

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight.data, 1)
                nn.init.constant_(layer.bias.data, 0)