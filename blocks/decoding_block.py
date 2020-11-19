from torch import nn

class DecodingBlock(nn.Module):
    """Decoding Block
    """
    def __init__(self, low_channels, in_channels, out_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, low_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True),
        )

        self.u = nn.Upsample(scale_factor=2, mode='bilinear')

        self.output_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False), # depthwise 3x3
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_x, x):
        return self.output_conv(low_x + self.u(self.input_conv(x)))
    
