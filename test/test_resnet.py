import unittest

import torch

from models import resnet


class MyTestCase(unittest.TestCase):
    def test_output(self):
        model = resnet.ResNet(c_channels=1,
                              classes=10,
                              block_sizes=[3,3,3],
                              feature_sizes=[16,16,32,64],
                              is_bottleneck=False,
                              initial_kernel_size=3,
                              skip_connections=True,
                              initial_stride=1,
                              initial_max_pool=False)
        inputs = torch.zeros((2,1,28,28))
        outputs = model(inputs)
        target_size = (2,10)
        conv_modules = [m for m in model.net.modules() if isinstance(m, torch.nn.Conv2d)]
        self.assertTrue(len(conv_modules) == 21)
        self.assertEqual(outputs.shape, target_size)

    def test_bottleneck(self):
        model = resnet.ResNet(c_channels=1,
                              classes=10,
                              block_sizes=[3,3,3],
                              feature_sizes=[16,16,32,64],
                              is_bottleneck=True,
                              initial_kernel_size=3,
                              skip_connections=True,
                              initial_stride=1,
                              initial_max_pool=False)
        inputs = torch.zeros((2,1,28,28))
        outputs = model(inputs)
        target_size = (2,10)
        conv_modules = [m for m in model.net.modules() if isinstance(m, torch.nn.Conv2d)]
        self.assertTrue(len(conv_modules) == 30)
        self.assertEqual(outputs.shape, target_size)

    def test_bottleneck_no_skip(self):
        model = resnet.ResNet(c_channels=1,
                              classes=10,
                              block_sizes=[3,3,3],
                              feature_sizes=[16,16,32,64],
                              is_bottleneck=True,
                              initial_kernel_size=3,
                              skip_connections=False,
                              initial_stride=1,
                              initial_max_pool=False)
        inputs = torch.zeros((2,1,28,28))
        outputs = model(inputs)
        target_size = (2,10)
        conv_modules = [m for m in model.net.modules() if isinstance(m, torch.nn.Conv2d)]
        self.assertTrue(len(conv_modules) == 28)
        #self.assertTrue(len(model.net._modules['3'].block) == 3)
        #self.assertTrue(len(model.net._modules['3'].block._modules['1'].net) == 9)
        self.assertEqual(outputs.shape, target_size)


if __name__ == '__main__':
    unittest.main()
