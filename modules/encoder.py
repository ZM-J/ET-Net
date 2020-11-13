from torch import nn
from args import ARGS 

# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py#L51
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, ARGS['encoder'][0], num_block[0], 1)
        self.conv3_x = self._make_layer(block, ARGS['encoder'][1], num_block[1], 2)
        self.conv4_x = self._make_layer(block, ARGS['encoder'][2], num_block[2], 2)
        self.conv5_x = self._make_layer(block, ARGS['encoder'][3], num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ARGS['encoder'][3] * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output_1 = self.conv2_x(output)
        output_2 = self.conv3_x(output)
        output_3 = self.conv4_x(output)
        output_4 = self.conv5_x(output)

        return output_1, output_2, output_3, output_4
