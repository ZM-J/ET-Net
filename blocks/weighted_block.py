from torch import nn

class WeightedBlock(nn.Module):
    """Weighted Block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_conv = self.input_conv(x)
        return input_conv * self.weight(input_conv)
    
