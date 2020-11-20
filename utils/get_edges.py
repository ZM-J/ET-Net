import os
from os import name
from PIL import Image
import numpy as np

def gradient(img):
    result_x = np.zeros_like(img, dtype=np.float)
    result_x[:, 0] = img[:, 1] - img[:, 0]
    result_x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    result_x[:, -1] = img[:, -1] - img[:, -2]
    
    result_y = np.zeros_like(img, dtype=np.float)
    result_y[0, :] = img[1, :] - img[0, :]
    result_y[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    result_y[-1, :] = img[-1, :] - img[-2, :]

    return result_x, result_y


def get_info(dataset_name):
    name2result = \
    {
        'DRIVE':
        {
            'train': {
                'folder_name': 'training',
                'id_list': [str(_) for _ in range(21, 37)],
            },
            'val': {
                'folder_name': 'val',
                'id_list': [str(_) for _ in range(37, 40)],
            },
            'label_folder': '1st_manual',
            'label_format': '{}_manual1.gif'
        },
        'CHASEDB1':
        {
            'train': {
                'folder_name': 'train',
                'id_list': [f'{num:02}{letter}' for num in range(1, 11) for letter in ['L', 'R']],
            },
            'val': {
                'folder_name': 'val',
                'id_list': [f'{num:02}{letter}' for num in range(11, 15) for letter in ['L', 'R']],
            },
            'label_folder': '1stHO',
            'label_format': 'Image_{}_1stHO.png'
        },
    }
    return name2result[dataset_name]


if __name__ == "__main__":
    dataset_name = 'CHASEDB1'
    info = get_info('CHASEDB1')
    for dataset_type in ['train', 'val']:
        dataset_folder = info[dataset_type]['folder_name']
        os.makedirs(f"../datasets/{dataset_name}/{dataset_folder}/edge", exist_ok=True)
        for img_id in info[dataset_type]['id_list']:
            in_name = f"../datasets/{dataset_name}/{dataset_folder}/{info['label_folder']}/{info['label_format']}".format(img_id)
            out_name = f"../datasets/{dataset_name}/{dataset_folder}/edge/{info['label_format']}".format(img_id)
            gt = np.array(Image.open(in_name).convert('L'))
            gx, gy = gradient(gt)
            temp_edge = gx * gx + gy * gy
            temp_edge[temp_edge != 0] = 1
            temp_edge = (temp_edge * 255).astype(np.uint8)
            Image.fromarray(temp_edge).save(out_name)