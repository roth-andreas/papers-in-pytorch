import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.contract1 = ContractingBlock(1, 64, pooling=False)
        self.contract2 = ContractingBlock(64, 128)
        self.contract3 = ContractingBlock(128, 256)
        self.contract4 = ContractingBlock(256, 512)
        self.contract5 = ContractingBlock(512, 1024)
        self.expand1 = ExpansiveBlock(1024, 512)
        self.expand2 = ExpansiveBlock(512, 256)
        self.expand3 = ExpansiveBlock(256, 128)
        self.expand4 = ExpansiveBlock(128, 64)
        self.output = nn.Conv2d(64, 1, 1, 1,0)
        self.init_weights()

    def forward(self, inputs):
        contracted1 = self.contract1(inputs)
        contracted2 = self.contract2(contracted1) 
        contracted3 = self.contract3(contracted2)
        contracted4 = self.contract4(contracted3)
        contracted5 = self.contract5(contracted4)

        expanded1 = self.expand1(contracted5, contracted4)
        expanded2 = self.expand2(expanded1, contracted3)
        expanded3 = self.expand3(expanded2, contracted2)
        expanded4 = self.expand4(expanded3, contracted1)

        return self.output(expanded4)

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