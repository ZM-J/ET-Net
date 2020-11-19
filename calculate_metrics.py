from torch.nn.parallel import DataParallel
from matplotlib import pyplot as plt
from model import ET_Net
from numpy.lib.twodim_base import mask_indices
import torch
from args import ARGS
from utils.get_dataset import get_dataset
import time
from torch.utils.data import DataLoader
from utils.crop_prediction import get_test_patches, recompone_overlap
from utils.metrics import calc_metrics
from PIL import Image
import os

class CalculateMetricProcess:
    def __init__(self):
        self.net = ET_Net()

        if (ARGS['gpu']):
            self.net = DataParallel(module=self.net.cuda())
        
        self.net.load_state_dict(torch.load(ARGS['weight']))

        self.metric_dataset = get_dataset(dataset_name=ARGS['dataset'], part='metric')

    def predict(self):

        start = time.time()
        self.net.eval()
        metric_dataloader = DataLoader(self.metric_dataset, batch_size=1) # only support batch size = 1
        os.makedirs(ARGS['prediction_save_folder'], exist_ok=True)
        y_true = []
        y_pred = []
        for items in metric_dataloader:
            images, labels, mask = items['image'], items['label'], items['mask']
            images = images.float()
            print('image shape:', images.size())

            image_patches, big_h, big_w = get_test_patches(images, ARGS['crop_size'], ARGS['stride_size'])
            test_patch_dataloader = DataLoader(image_patches, batch_size=ARGS['batch_size'], shuffle=False, drop_last=False)
            test_results = []
            print('Number of batches for testing:', len(test_patch_dataloader))

            for patches in test_patch_dataloader:
                
                if ARGS['gpu']:
                    patches = patches.cuda()
                
                with torch.no_grad():
                    result_patches_edge, result_patches = self.net(patches)
                
                test_results.append(result_patches.cpu())           
            
            test_results = torch.cat(test_results, dim=0)
            # merge
            test_results = recompone_overlap(test_results, ARGS['crop_size'], ARGS['stride_size'], big_h, big_w)
            test_results = test_results[:, 1, :images.size(2), :images.size(3)]
            y_pred.append(test_results[mask == 1].reshape(-1))
            y_true.append(labels[mask == 1].reshape(-1))
        
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        calc_metrics(y_pred, y_true)
        finish = time.time()

        print('Calculating metric time consumed: {:.2f}s'.format(finish - start))

if __name__ == "__main__":
    cmp = CalculateMetricProcess()
    cmp.predict()