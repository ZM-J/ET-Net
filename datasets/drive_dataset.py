import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from args import ARGS
from datasets.data_augmentation import data_augmentation, _random_crop

class DRIVEDataset(Dataset):
    def __init__(self, data_path, label_path, edge_path, need_enhance=True):
        super(DRIVEDataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.edge_path = edge_path
        
        self.need_enhance = need_enhance
        self.data_list = [_ for _ in os.listdir(data_path) if '.tif' in _]
        self.label_list = [_ for _ in os.listdir(label_path) if '.gif' in _]
        self.edge_list = [_ for _ in os.listdir(edge_path) if '.gif' in _]
        self.data_id = [_.split('_')[0] for _ in self.data_list]
        self.label_id = [_.split('_')[0] for _ in self.data_list]
        assert len(self.data_list) == len(self.label_list), \
            f"The number of data ({len(self.data_list)}) doesn't match the number of labels ({len(self.label_list)})"
        # Try to match data_list and label_list
        assert set(self.data_id) == set(self.label_id)

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, f'{self.data_id[index]}_training.tif'))
        label = Image.open(os.path.join(self.label_path, f'{self.data_id[index]}_manual1.gif'))
        edge = Image.open(os.path.join(self.edge_path, f'{self.data_id[index]}_manual1.gif'))

        # Data Augmentation
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(scale=(0.8, 1.2)), # 0.5, 2
        # ])
        if (self.need_enhance):
            img, label, edge = data_augmentation(img, label, edge)
            
        img, label, edge = _random_crop(img, label, edge)

        label = label.convert('1')
        edge = edge.convert('1')

        img = np.array(img).transpose((2, 0, 1)) / 255. # [0, 255] -> [0, 1]
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)

        label = torch.LongTensor(np.array(label)) // 255 # 512 x 512 # [0, 255] -> [0, 1]
        edge = torch.LongTensor(np.array(edge)) // 255 # 512 x 512
        
        return {'image': img, 'label': label, 'edge': edge}
    
    def __len__(self) -> int:
        return len(self.data_list)


class DRIVETestDataset(Dataset):
    def __init__(self, data_path, mask_path):
        super(DRIVETestDataset, self).__init__()
        self.data_path = data_path
        self.mask_path = mask_path # mask is optional
        self.data_list = [_ for _ in os.listdir(data_path) if '.tif' in _]
        self.mask_list = [_ for _ in os.listdir(mask_path) if '.gif' in _]
        self.data_id = [_.split('_')[0] for _ in self.data_list]
        self.mask_id = [_.split('_')[0] for _ in self.mask_list]
        assert len(self.data_list) == len(self.mask_list), \
            f"The number of data ({len(self.data_list)}) doesn't match the number of masks ({len(self.mask_list)})"
        # Try to match data_list and label_list
        assert set(self.data_id) == set(self.mask_id)

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, f'{self.data_id[index]}_test.tif'))
        mask = Image.open(os.path.join(self.mask_path, f'{self.data_id[index]}_test_mask.gif'))
        
        img = np.array(img).transpose((2, 0, 1)) / 255.
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)
        mask = torch.LongTensor(np.array(mask)) // 255

        return {'image': img, 'mask': mask, 'filename': self.data_list[index]}
    
    def __len__(self) -> int:
        return len(self.data_list)

class DRIVEMetricDataset(DRIVEDataset):
    def __init__(self, data_path, label_path, edge_path, mask_path):
        super(DRIVEMetricDataset, self).__init__(data_path, label_path, edge_path, False)
        self.mask_path = mask_path # mask is optional
        self.mask_list = [_ for _ in os.listdir(mask_path) if '.gif' in _]

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, f'{self.data_id[index]}_training.tif'))
        label = Image.open(os.path.join(self.label_path, f'{self.data_id[index]}_manual1.gif'))
        edge = Image.open(os.path.join(self.edge_path, f'{self.data_id[index]}_manual1.gif'))
        mask = Image.open(os.path.join(self.mask_path, f'{self.data_id[index]}_training_mask.gif'))
        label = label.convert('1')
        edge = edge.convert('1')        
        mask = mask.convert('1')

        img = np.array(img).transpose((2, 0, 1)) / 255. # [0, 255] -> [0, 1]
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)

        label = torch.LongTensor(np.array(label)) // 255 # 512 x 512 # [0, 255] -> [0, 1]
        edge = torch.LongTensor(np.array(edge)) // 255 # 512 x 512
        mask = torch.LongTensor(np.array(mask)) // 255 # 512 x 512
        return {'image': img, 'label': label, 'edge': edge, 'mask': mask}
    
    def __len__(self) -> int:
        return len(self.data_list)