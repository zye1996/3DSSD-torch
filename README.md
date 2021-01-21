# 3DSSD-pytorch
 3DSSD's implementation with Pytorch

This repository contains a PyTorch implementation of [3DSSD](https://github.com/Jia-Research-Lab/3DSSD) on the KITTI benchmark.

There are several characteristics to make you easy to understand and modify the code:
1. I keep the name of the folders. files and the fucntions the same as the [official code](https://github.com/Jia-Research-Lab/3DSSD) as much as possible. 
2. The "Trainner" in the lib/core/trainner.py draws on the code style of the [PCDet](https://github.com/open-mmlab/OpenPCDet).
3. I borrow the visualization code with the MeshLab from the [VoteNet](https://github.com/facebookresearch/votenet).

If you want to use the F-FPS or the Dilated-Ball-Query, you can just use the files in lib/pointnet2.

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


