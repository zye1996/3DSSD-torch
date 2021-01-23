# 3DSSD-torch

```diff
- WARNING: This is a forked version from 3DSSD-pytorch since I cannot reach the author and the original version is broken. 
- I move the repo here and try to fix the bugs. 
- Some progresses have been made but problems persist in low mAP score
```

3DSSD's implementation with Pytorch

This repository contains a PyTorch implementation of [3DSSD](https://github.com/Jia-Research-Lab/3DSSD) on the KITTI benchmark.

There are several characteristics to make you easy to understand and modify the code:
1. I keep the name of the folders. files and the fucntions the same as the [official code](https://github.com/Jia-Research-Lab/3DSSD) as much as possible. 
2. The "Trainner" in the lib/core/trainner.py draws on the code style of the [PCDet](https://github.com/open-mmlab/OpenPCDet).
3. I borrow the visualization code with the MeshLab from the [VoteNet](https://github.com/facebookresearch/votenet).

## System Requirement

Test with following configuration:

PyTorch = 1.7.1

Cuda = 11.1

numba = 0.48

## Preparation

1. Clone this repository

2. Install the Python dependencies.

```
pip install -r requirements.txt
```

3. Install python functions. the functions are partly borrowed from the Pointnet2 in [PointRCNN](https://github.com/sshaoshuai/PointRCNN). The F-FPS and dilated-ball-query are implemented by myself.

```
cd lib/pointnet2
python setup.py install
```

4. Prepare data with according to the "Data Preparation" in the [3DSSD](https://github.com/Jia-Research-Lab/3DSSD)

## Train a Model

```
python lib/core/trainer.py --cfg configs/kitti/3dssd/3dssd.yaml
```

The trainning log and tensorboard log are saved into output dir

## Performance
I get the result as follow but still some gaps compared to the original version. I will continue to debug the code.

```
bbox AP:88.3067, 87.4015, 87.4015
bev  AP:87.1555, 83.6703, 83.6703
3d   AP:81.8364, 75.6352, 75.6352
aos  AP:88.31, 87.40, 87.40
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:92.6265, 89.2507, 89.2507
bev  AP:89.0312, 84.9429, 84.9429
3d   AP:83.2703, 74.8559, 74.8559
aos  AP:92.63, 89.25, 89.25
Car AP@0.70, 0.50, 0.50:
bbox AP:88.3067, 87.4015, 87.4015
bev  AP:88.4671, 87.8447, 87.8447
3d   AP:88.4409, 87.7542, 87.7542
aos  AP:88.31, 87.40, 87.40
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:92.6265, 89.2507, 89.2507
bev  AP:92.9291, 91.6493, 91.6493
3d   AP:92.8714, 91.3892, 91.3892
aos  AP:92.63, 89.25, 89.25
```
