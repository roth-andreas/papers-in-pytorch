import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    '''
    Implementation of the U-Net Model (https://arxiv.org/abs/1505.04597)
    '''
    def __init__(self, filter_sizes=[64,128,256,512,1024],
                 skip_connections=True,
                 padding=0,
                c_channels=1,
                upsampling=None):
        ''' Initializes the UNet Model
        
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
        '''
        super(UNet, self).__init__()
        self.contractors = nn.ModuleList([ContractingBlock(c_channels, filter_sizes[0], pooling=False, padding=padding)])
        self.expanders = nn.ModuleList()
        for index in range(len(filter_sizes) - 1):
            self.contractors.append(ContractingBlock(filter_sizes[index], filter_sizes[index + 1], padding=padding))
            self.expanders.append(ExpansiveBlock(filter_sizes[-(index+1)], filter_sizes[-(index+2)], skip_connections, padding=padding, upsampling=upsampling))
        self.output = nn.Conv2d(filter_sizes[0], 1, 1, 1,0)
        self.init_weights()

    def forward(self, inputs):
        outputs = [self.contractors[0](inputs)]
        for layer in self.contractors[1:]:
            #print(outputs[-1].size())
            outputs += [layer(outputs[-1])]
        for index, layer in enumerate(self.expanders):
            #print(outputs[-1].size())
            outputs += [layer(outputs[-1], outputs[len(self.expanders) - index - 1])]

        return self.output(outputs[-1])

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias.data, 0)

class ExpansiveBlock(nn.Module):
    # Defines one expansive Block of the U-Net
    
    def __init__(self, in_dim, out_dim, skip_connections, padding=0, upsampling=None):
        super(ExpansiveBlock, self).__init__()
        if upsampling is None:
            self.upsample = nn.ConvTranspose2d(in_dim, out_dim, 2, 2, 0)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ZeroPad2d([0,1,0,1]),
                nn.Conv2d(in_dim, out_dim, 2, 1,0)
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_dim if skip_connections else out_dim, out_dim, 3, 1,padding),
            nn.ReLU(True),
            nn.Conv2d(out_dim, out_dim,3,1,padding),
            nn.ReLU(True)
        )
        self.skip_connections = skip_connections
        self.upsampling = upsampling

    def crop_center(self, inputs, target_size):
        start_crop = (inputs.size(2) - target_size) // 2
        return inputs[:,:,start_crop:start_crop+target_size,start_crop:start_crop+target_size]
  
    def forward(self, inputs, skip_connected):
        x = self.upsample(inputs)

        if self.skip_connections:
            skip_connected = self.crop_center(skip_connected, x.size(3))
            x = torch.cat((x, skip_connected), dim=1)
        
        return self.block(x)

class ContractingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pooling=True, padding=0):
        super(ContractingBlock, self).__init__()
        layers = []
        if pooling:
            layers += [nn.MaxPool2d(2,2)]

        layers += [
              nn.Conv2d(in_dim, out_dim, 3, 1, padding),
              nn.ReLU(True),
              nn.Conv2d(out_dim, out_dim, 3, 1, padding),
              nn.ReLU(True)
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)