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
from PIL import Image
import os

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
        test_dataloader = DataLoader(self.test_dataset, batch_size=1) # only support batch size = 1
        os.makedirs(ARGS['prediction_save_folder'], exist_ok=True)
        for items in test_dataloader:
            images, mask, filename = items['image'], items['mask'], items['filename']
            images = images.float()
            mask = mask.long()
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
            test_results = test_results[:, 1, :images.size(2), :images.size(3)] * mask
            test_results = Image.fromarray(test_results[0].numpy())
            test_results.save(os.path.join(ARGS['prediction_save_folder'], filename[0]))
            print(f'Finish prediction for {filename[0]}')

        finish = time.time()

        print('Predicting time consumed: {:.2f}s'.format(finish - start))

if __name__ == "__main__":
    tp = TestProcess()
    tp.predict()