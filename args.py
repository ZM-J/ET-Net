
ARGS = {
    'encoder': [64, 128, 256, 512],
    'decoder': [256, 128, 64],
    'egm': [2, 64],
    'wam': [64, 3],
    'encoder_weight': r"../resnet50_weight/resnet50-19c8e357.pth",
    'gpu': True,
    'weight': None,
    # 'weight': '../weights/xx.pth'
    'dataset': 'DRIVE',
    'num_epochs': 300,
    'batch_size': 16,
    'lr': 5e-3,
    'scheduler_power': 0.9,
    'combine_alpha': 0.3,
    'save_folder': 'weights'
}