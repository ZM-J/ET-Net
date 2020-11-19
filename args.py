
ARGS = {
    'encoder': [64, 128, 256, 512],
    'decoder': [256, 128, 64],
    'egm': [2, 64],
    'wam': [64, 2],
    'crop_size': 128,
    'stride_size': 7,
    'encoder_weight': "../resnet50_weight/resnet50-19c8e357.pth",
    'gpu': True,
    # 'weight': None,
    'weight': 'weights/epoch_270.pth', # weights
    # 'weight': 'weights_/epoch_300.pth', # weights
    'dataset': 'DRIVE',
    'num_epochs': 300,
    'epoch_save': 30,
    'batch_size': 4,
    'lr': 5e-3,
    'scheduler_power': 0.9,
    'combine_alpha': 0.3,
    'weight_save_folder': 'weights',
    'prediction_save_folder': 'test_results'
}