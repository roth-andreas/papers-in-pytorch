import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, classes=1000, c_channels=3, dropout=0.5, device=None):
        super(VGG, self).__init__()
        self.convs = nn.Sequential(
            ConvBlock(1, c_channels, 64),
            # 112 x 112
            ConvBlock(1, 64, 128),
            # 56 x 56
            ConvBlock(2, 128, 256),
            # 28 x 28
            ConvBlock(2, 256, 512),
            # 14 x 14
            ConvBlock(2, 512, 512)
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
        x = self.convs(inputs):
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x
        
        
class ConvBlock(nn.Module):
    def __init__(self, depth, input_width, output_width):
        super(ConvBlock, self).__init__()
        layers = []
        layers += [nn.Conv2d(input_width, output_width, 3, 1, 1), nn.ReLU(True)]
        for i in range(1, depth):
            layers += [nn.Conv2d(output_width, output_width, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.MaxPool2d(2, 2)]
        self.net = nn.Sequential(
            *layers
        )
        
    def forward(self, inputs):
        return self.net(inputs)