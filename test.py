from torch.nn.parallel import DataParallel
from matplotlib import pyplot as plt
from model import ET_Net
from numpy.lib.twodim_base import mask_indices
import torch
from args import ARGS
from get_dataset import get_dataset
import time
from torch.utils.data import DataLoader
from crop_prediction import paint_border

class TestProcess:
    def __init__(self):
        self.net = ET_Net()

        if (ARGS['gpu']):
            self.net = DataParallel(module=self.net.cuda())
        
        self.net.load_state_dict(torch.load(ARGS['weight']))

        self.test_dataset = get_dataset(dataset_name=ARGS['dataset'], part='test')

    def predict(self):

        start = time.time()
        self.net.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size=1)
        for batch_index, items in enumerate(test_dataloader):
            images, filename = items['image'], items['filename']
            images = images.float()
            images = paint_border(images)
            raise Exception

            if ARGS['gpu']:
                labels = labels.cuda()
                images = images.cuda()
                edges = edges.cuda()
            
            print('image shape:', images.size())

            with torch.no_grad():
                outputs_edge, outputs = self.net(images)
            
            pred = torch.max(outputs, dim=1)[1]
            iou = torch.sum(pred[0] & labels[0]) / (torch.sum(pred[0] | labels[0]) + 1e-6)
            
            mean = torch.FloatTensor([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
            images = images + mean.cuda()

            # images *= 255.
            print('pred min: ', pred[0].min(), ' max: ', pred[0].max())
            print('label min:', labels[0].min(), ' max: ', labels[0].max())
            print('edge min:', edges[0].min(), ' max: ', edges[0].max())
            print('output edge min:', outputs_edge[0].min(), ' max: ', outputs_edge[0].max())
            print('IoU:', iou)
            print('Intersect num:', torch.sum(pred[0] & labels[0]))
            print('Union num:', torch.sum(pred[0] | labels[0]))
            

            plt.subplot(221)
            plt.imshow(images[0].cpu().numpy().transpose((1, 2, 0))), plt.axis('off')
            plt.subplot(222)
            plt.imshow(labels[0].cpu().numpy(), cmap='gray'), plt.axis('off')
            plt.subplot(223)
            # plt.imshow(pred[0].cpu().numpy(), cmap='gray'), plt.axis('off')
            plt.imshow(outputs[0, 1].cpu().numpy(), cmap='gray'), plt.axis('off')
            plt.subplot(224)
            plt.imshow(outputs_edge[0, 1].cpu().numpy(), cmap='gray'), plt.axis('off')
            plt.show()

            

            # update training loss for each iteration
            # self.writer.add_scalar('Train/loss', loss.item(), n_iter)

        finish = time.time()

        print('validating time consumed: {:.2f}s'.format(finish - start))

if __name__ == "__main__":
    tp = TestProcess()
    tp.predict()