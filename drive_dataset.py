from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import os
import random

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
            img, label, edge = DRIVEDataset._random_mirror(img, label, edge)
            img, label, edge = DRIVEDataset._random_scale(img, label, edge)
            img, label, edge = DRIVEDataset._random_rotation(img, label, edge)
            img = DRIVEDataset._random_color_jitter(img)
            img, label, edge = DRIVEDataset._random_crop(img, label, edge)

        label = label.convert('1')
        edge = edge.convert('1')

        img = np.array(img)
        label = np.array(label)
        edge = np.array(edge)
        return {'image': img, 'label': label, 'edge': edge}
    
    def __len__(self) -> int:
        return len(self.data_list)

    def _random_mirror(img, label, edge):
        r_img, r_label, r_edge = img, label, edge
        if random.random() < 0.5:
            r_img = r_img.transpose(Image.FLIP_LEFT_RIGHT)
            r_label = r_label.transpose(Image.FLIP_LEFT_RIGHT)
            r_edge = r_edge.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            r_img = r_img.transpose(Image.FLIP_TOP_BOTTOM)
            r_label = r_label.transpose(Image.FLIP_TOP_BOTTOM)
            r_edge = r_edge.transpose(Image.FLIP_TOP_BOTTOM)
        return r_img, r_label, r_edge
    
    def _random_scale(img, label, edge):
        r_img, r_label, r_edge = img, label, edge
        if np.random.random() < 0.05:
            z = np.random.uniform(0.8, 1.2) # 0.5 ~ 2
            width, height = img.size
            to_width, to_height = int(z*width), int(z*height)
            r_img = img.resize((to_width, to_height), Image.ANTIALIAS)
            r_label = label.resize((to_width, to_height), Image.ANTIALIAS)
            r_edge = edge.resize((to_width, to_height), Image.ANTIALIAS)
        return r_img, r_label, r_edge
    
    def _random_rotation(img, label, edge):
        r_img, r_label, r_edge = img, label, edge
        if np.random.random() < 0.5:
            theta = np.random.uniform(-10, 10)
            r_img = img.rotate(theta)
            r_label = label.rotate(theta)
            r_edge = edge.rotate(theta)
        return r_img, r_label, r_edge
    
    def _random_color_jitter(img):
        r_img = img
        transform_tuples = [
            (ImageEnhance.Brightness, 0.1026),
            (ImageEnhance.Contrast, 0.0935),
            (ImageEnhance.Sharpness, 0.8386),
            (ImageEnhance.Color, 0.1592)
        ]
        if np.random.random() < 0.5:
            rand_num = np.random.uniform(0, 1, len(transform_tuples))
            for i, (transformer, alpha) in enumerate(transform_tuples):
                r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
                r_img = transformer(r_img).enhance(r)
        return r_img
    
    def _random_crop(img, label, edge):
        r_img, r_label, r_edge = img, label, edge
        width, height = img.size
        r_width, r_height = 512, 512
        zx, zy = random.randint(0, width - r_width - 1), random.randint(0, height - r_height - 1)
        r_img = r_img.crop((zx, zy, zx+r_width, zy+r_height))
        r_label = r_label.crop((zx, zy, zx+r_width, zy+r_height))
        r_edge = r_edge.crop((zx, zy, zx+r_width, zy+r_height))
        return r_img, r_label, r_edge


class DRIVETestDataset(Dataset):
    def __init__(self, data_path, need_enhance=False):
        super(DRIVETestDataset, self).__init__()
        self.data_path = data_path
        self.need_enhance = need_enhance
        self.data_list = [_ for _ in os.listdir(data_path) if '.tif' in _]

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, self.data_list[index]))

        img = np.array(img)
        return {'image': img}
    
    def __len__(self) -> int:
        return len(self.data_list)