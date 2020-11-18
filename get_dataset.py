from drive_dataset import DRIVEDataset, DRIVETestDataset

def get_dataset(dataset_name, part):
    dataset_dict = {
        'DRIVE': {
            'train': DRIVEDataset(
                data_path='../datasets/DRIVE/training/images',
                label_path='../datasets/DRIVE/training/1st_manual',
                edge_path='../datasets/DRIVE/training/edge',
                need_enhance=True
            ),
            'val': DRIVEDataset(
                data_path='../datasets/DRIVE/val/images',
                label_path='../datasets/DRIVE/val/1st_manual',
                edge_path='../datasets/DRIVE/val/edge',
                need_enhance=False
            ),
            'test': DRIVETestDataset( # TODO
                data_path='../datasets/DRIVE/test/images',
                need_enhance=False
            )
        }
    }
    return dataset_dict[dataset_name][part]