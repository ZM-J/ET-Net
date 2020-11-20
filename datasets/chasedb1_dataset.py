import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from args import ARGS
from datasets.data_augmentation import data_augmentation, _random_crop

class ChaseDB1Dataset(Dataset):
    def __init__(self, data_path, label_path, edge_path, need_enhance=True):
        super(ChaseDB1Dataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.edge_path = edge_path
        
        self.need_enhance = need_enhance
        self.data_list = [_ for _ in os.listdir(data_path) if '.jpg' in _]
        self.label_list = [_ for _ in os.listdir(label_path) if '.png' in _]
        self.edge_list = [_ for _ in os.listdir(edge_path) if '.png' in _]
        self.data_id = [_[6:9] for _ in self.data_list]
        self.label_id = [_[6:9] for _ in self.label_list]
        assert len(self.data_list) == len(self.label_list), \
            f"The number of data ({len(self.data_list)}) doesn't match the number of labels ({len(self.label_list)})"
        # Try to match data_list and label_list
        assert set(self.data_id) == set(self.label_id)

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, f'Image_{self.data_id[index]}.jpg'))
        label = Image.open(os.path.join(self.label_path, f'Image_{self.data_id[index]}_1stHO.png'))
        edge = Image.open(os.path.join(self.edge_path, f'Image_{self.data_id[index]}_1stHO.png'))

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


class ChaseDB1TestDataset(Dataset):
    def __init__(self, data_path):
        super(ChaseDB1TestDataset, self).__init__()
        self.data_path = data_path
        self.data_list = [_ for _ in os.listdir(data_path) if '.jpg' in _]
        self.data_id = [_[6:9] for _ in self.data_list]

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, f'Image_{self.data_id[index]}.jpg'))
        
        img = np.array(img).transpose((2, 0, 1)) / 255.
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)
        img_c, img_h, img_w = img.size()
        mask = torch.ones((img_h, img_w)).long()

        return {'image': img, 'mask': mask, 'filename': self.data_list[index]}
    
    def __len__(self) -> int:
        return len(self.data_list)

class ChaseDB1MetricDataset(ChaseDB1Dataset):
    def __init__(self, data_path, label_path, edge_path):
        super(ChaseDB1MetricDataset, self).__init__(data_path, label_path, edge_path, False)

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, f'Image_{self.data_id[index]}.jpg'))
        label = Image.open(os.path.join(self.label_path, f'Image_{self.data_id[index]}_1stHO.png'))
        edge = Image.open(os.path.join(self.edge_path, f'Image_{self.data_id[index]}_1stHO.png'))
        label = label.convert('1')
        edge = edge.convert('1')        

        img = np.array(img).transpose((2, 0, 1)) / 255. # [0, 255] -> [0, 1]
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)

        label = torch.LongTensor(np.array(label)) // 255 # 512 x 512 # [0, 255] -> [0, 1]
        edge = torch.LongTensor(np.array(edge)) // 255 # 512 x 512

        img_c, img_h, img_w = img.size()
        mask = torch.ones((img_h, img_w)).long()
        return {'image': img, 'label': label, 'edge': edge, 'mask': mask}
    
    def __len__(self) -> int:
        return len(self.data_list)