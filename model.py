import torch
from torch import nn
from modules.encoder import Encoder

from modules.decoder import Decoder
from modules.edge_guidance_module import EdgeGuidanceModule
from modules.weighted_aggregation_module import WeightedAggregationModule
from args import ARGS


class ET_Net(nn.Module):
    """ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.egm = EdgeGuidanceModule()
        self.wam = WeightedAggregationModule()

    def forward(self, x):
        enc_1, enc_2, enc_3, enc_4 = self.encoder(x)
        dec_1, dec_2, dec_3 = self.decoder(enc_1, enc_2, enc_3, enc_4)
        edge_pred, egm = self.egm(enc_1, enc_2)
        pred = self.wam(dec_1, dec_2, dec_3, egm)
        return edge_pred, pred
    
    def load_encoder_weight(self):
        # One could get the pretrained weights via PyTorch official.
        self.encoder.load_state_dict(torch.load(ARGS['encoder_weight']))

if __name__ == "__main__":
    net = ET_Net()
    net.load_encoder_weight()
    
    net = net.cuda()
    net.train()
    img = torch.randn((2, 3, 512, 512)).cuda()
    out_edge, out = net(img)
    print(len(out))
    print(len(out[0]))
    print(out[0].shape)
    print(out_edge[0].shape)
    # print(net)
    input('Press Enter to Continue...')