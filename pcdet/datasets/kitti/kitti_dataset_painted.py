import copy
import pickle

import numpy as np
from skimage import io

from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from .kitti_dataset import KittiDataset


class KittiDatasetPainted(KittiDataset):

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne_painted_mono' / ('%s.npy' % idx)
        assert lidar_file.exists()
        data = np.load(lidar_file)
        return data


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=8):
    dataset = KittiDatasetPainted(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)

    print('---------------Start create painted groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
