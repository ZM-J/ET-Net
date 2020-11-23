# ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation
---
## Accepted by MICCAI 2019
---
A non-official simple PyTorch implementation of ET-Net

The difference between this implementation and the paper is that, this implementation uses 128 x 128 patches to train the network and make predictions, and then use sliding window to merge predictions. This is because:

i. when I try to use 512 x 512 cropped patch with batch size = 16, my GPU will be out of memory (because neither code nor detailed network setting has been released by the authors, this setting might be different from what the authors use)

ii. model prediction (segmentation) will be more robust if sliding window technique is applied

## Quick start

1. Change the contents of `utils/get_dataset.py` to your directories

2. Run `utils/get_edges.py` to get edges of labels in your dataset

3. Run `utils/get_pretrained_weight_address.py` to get pretrained weights for ResNet50 (E-Blocks)

4. Change settings in `args.py`, including dataset chosen to train, batch size, etc.

5. Run `train.py` to train ET-Net on some dataset

6. Run `visualize.py` to observe input, label, prediction, and edge prediction

7. Run `calculate_metrics.py` to get metrics (AUC, accuracy, mIoU) on validate dataset

8. Run `test.py` to predict segmentation results using ET-Net

## Results

metric|DRIVE|CHASE-DB1|MC|LUNA
---|---|---|---|---
AUC|96.25|96.49|???|???
Acc.|95.35|97.07|???|???
mIoU|68.14|61.19|???|???

##  TODO

Implement more datasets and get results from these datasets