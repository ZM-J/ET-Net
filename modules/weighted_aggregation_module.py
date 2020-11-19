import torch
from torch import nn
from blocks.weighted_block import WeightedBlock
from torchvision.models.resnet import Bottleneck
from args import ARGS

class WeightedAggregationModule(nn.Module):
    """Weighted Aggregation Module
    """
    def __init__(self):
        super().__init__()
        in_1_channels = ARGS['decoder'][0] * Bottleneck.expansion
        in_2_channels = ARGS['decoder'][1] * Bottleneck.expansion
        in_3_channels = ARGS['decoder'][2] * Bottleneck.expansion
        in_e_channels = ARGS['egm'][1] * Bottleneck.expansion
        out_1_channels = ARGS['wam'][0] * Bottleneck.expansion
        out_channels = ARGS['wam'][1]

        self.weight_1 = nn.Sequential(
            WeightedBlock(in_1_channels, out_1_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.weight_2 = WeightedBlock(in_2_channels, out_1_channels)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.weight_3 = WeightedBlock(in_3_channels, out_1_channels)
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_1_channels + in_e_channels, out_channels, kernel_size=1), # 1x1
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x_1, x_2, x_3, x_e):
        weight_1 = self.weight_1(x_1)
        weight_2 = self.up_2(weight_1 + self.weight_2(x_2))
        weight_3 = weight_2 + self.weight_3(x_3)
        weight_c = torch.cat((weight_3, x_e), dim=1)
        return self.output_conv(weight_c)