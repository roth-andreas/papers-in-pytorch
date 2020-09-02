import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, feature_sizes=[64,128,256,512,1024]):
        super(UNet, self).__init__()
        self.contractors = nn.ModuleList([ContractingBlock(1, feature_sizes[0], pooling=False)])
        self.expanders = nn.ModuleList()
        for index in range(len(feature_sizes) - 1):
            self.contractors.append(ContractingBlock(index, index + 1))
            self.expanders.append(ExpansiveBlock(feature_sizes[-i], feature_sizes[-(i+1)]))
        self.output = nn.Conv2d(64, 1, 1, 1,0)
        self.init_weights()

    def forward(self, inputs):
        outputs = [self.contractors[0](inputs)]
        for layer in self.contractors:
            outputs += [layer(outputs[-1])]

        for index, layer in enumerate(self.expanders):
            outputs += [layer(outputs[-1], outputs[len(self.expanders) - i - 1])]

        return self.output(outputs[-1])

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias.data, 0)

class ExpansiveBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ExpansiveBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_dim, out_dim, 2, 2, 0)
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1,0),
            nn.ReLU(True),
            nn.Conv2d(out_dim, out_dim,3,1,0),
            nn.ReLU(True)
        )

    def crop_center(self, inputs, target_size):
        start_crop = (inputs.size(2) - target_size) // 2
        return inputs[:,:,start_crop:start_crop+target_size,start_crop:start_crop+target_size]
  
    def forward(self, inputs, skip_connected):
        x = self.upconv(inputs)
        skip_connected = self.crop_center(skip_connected, x.size(3))

        x = torch.cat((x, skip_connected), dim=1)
        return self.block(x)

class ContractingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pooling=True):
        super(ContractingBlock, self).__init__()
        layers = []
        if pooling:
            layers += [nn.MaxPool2d(2,2)]

        layers += [
              nn.Conv2d(in_dim, out_dim, 3, 1, 0),
              nn.ReLU(True),
              nn.Conv2d(out_dim, out_dim, 3, 1, 0),
              nn.ReLU(True)
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)