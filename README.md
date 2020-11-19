# ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation
---
## Accepted by MICCAI 2019
---
A non-official PyTorch implementation of ET-Net

## Quick start

1. Change the contents of `get_dataset.py` to your directories

2. Run `get_pretrained_weight_address.py` to get pretrained weights for ResNet50 (E-Blocks)

3. Change settings in `args.py`, including dataset chosen to train, batch size, etc.

4. Run `main.py` and set `state = 'train'` to train ET-Net on some dataset

5. Run `main.py` and set `state = 'test'` to predict segmentation results using ET-Net 