from torchvision.models.resnet import ResNet, Bottleneck
from torch import nn

class Encoder(ResNet):
    def __init__(self):
        super(Encoder, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3])
        self.conv1.stride = 1 # 2 -> 1: No size/2

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # No maxpool: No size/2

        output_1 = self.layer1(x)
        output_2 = self.layer2(output_1)
        output_3 = self.layer3(output_2)
        output_4 = self.layer4(output_3)

        return output_1, output_2, output_3, output_4

