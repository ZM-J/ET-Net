from datasets.drive_dataset import DRIVEDataset, DRIVETestDataset, DRIVEMetricDataset

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
            'test': DRIVETestDataset(
                data_path='../datasets/DRIVE/test/images',
                mask_path='../datasets/DRIVE/test/mask',
            ),
            'metric': DRIVEMetricDataset(
                data_path='../datasets/DRIVE/val/images',
                label_path='../datasets/DRIVE/val/1st_manual',
                edge_path='../datasets/DRIVE/val/edge',
                mask_path='../datasets/DRIVE/val/mask',
            ),
        }
    }
    return dataset_dict[dataset_name][part]