from torchvision.models.resnet import ResNet, Bottleneck

class Encoder(ResNet):
    def __init__(self):
        super(Encoder, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output_1 = self.layer1(x)
        output_2 = self.layer2(x)
        output_3 = self.layer3(x)
        output_4 = self.layer4(x)

        return output_1, output_2, output_3, output_4

