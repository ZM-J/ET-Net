from datasets.drive_dataset import DRIVEDataset, DRIVETestDataset, DRIVEMetricDataset
from datasets.chasedb1_dataset import ChaseDB1Dataset, ChaseDB1TestDataset, ChaseDB1MetricDataset

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
        },
        'CHASEDB1': {
            'train': ChaseDB1Dataset(
                data_path='../datasets/CHASEDB1/train/images',
                label_path='../datasets/CHASEDB1/train/1stHO',
                edge_path='../datasets/CHASEDB1/train/edge',
                need_enhance=True
            ),
            'val': ChaseDB1Dataset(
                data_path='../datasets/CHASEDB1/val/images',
                label_path='../datasets/CHASEDB1/val/1stHO',
                edge_path='../datasets/CHASEDB1/val/edge',
                need_enhance=False
            ),
            # 'test': ChaseDB1TestDataset(
            #     data_path='../datasets/CHASEDB1/test/images',
            # ),
            'metric': ChaseDB1MetricDataset(
                data_path='../datasets/CHASEDB1/val/images',
                label_path='../datasets/CHASEDB1/val/1stHO',
                edge_path='../datasets/CHASEDB1/val/edge',
            ),
        }
    }
    return dataset_dict[dataset_name][part]