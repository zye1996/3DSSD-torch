import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
from kitti_util import *
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'raw_points': points,
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class Detector:

    def __init__(self, cfg, ckpt_file, calib_file, demo_dataset):
        self.logger = common_utils.create_logger()
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        self.model.load_params_from_file(filename=ckpt_file, logger=self.logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()
        self.calib = Calibration(calib_file, from_video=True)

    def run(self, data_dict, threshold=0.5):
        # feed forward
        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        obj_ind = pred_dicts[0]['pred_scores'] >= threshold
        pred_3d = pred_dicts[0]['pred_boxes'][obj_ind, ...].cpu().numpy()
        pred_3d = boxes_to_corners_3d(pred_3d)
        pred_2d = []
        for obj in pred_3d:
            obj_2d = self.calib.project_velo_to_image(obj)
            pred_2d.append([np.min(obj_2d, axis=0)[0], np.min(obj_2d, axis=0)[1],
                            np.max(obj_2d, axis=0)[0], np.max(obj_2d, axis=0)[1]])
        return pred_2d, [xx.squeeze() for xx in np.split(pred_3d, pred_3d.shape[0], axis=0)]


if __name__ == "__main__":

    cfg_from_yaml_file("/tools/cfgs/kitti_models/3dssd_diou.yaml", cfg)
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path("/home/yzy/Documents/dataset/kitti/raw/2011_09_26/2011_09_26_drive_0023_sync/velodyne_points/data"),
        ext=".bin")
    detector = Detector(cfg,
                        ckpt_file="/home/yzy/PycharmProjects/3DSSD-torch/output/kitti_models/3dssd/default/ckpt/checkpoint_epoch_120.pth",
                        calib_file="/home/yzy/Documents/dataset/kitti/raw/2011_09_26",
                        demo_dataset=demo_dataset)

    data_dict = demo_dataset[0]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    pred_2d, pred_3d = detector.run(data_dict, threshold=0.2)

    image = cv2.imread("/home/yzy/Documents/dataset/kitti/raw/2011_09_26/2011_09_26_drive_0023_sync/image_02/data/0000000000.png")

    for box in pred_2d:
        top_left = int(box[0]), int(box[1])
        bottom_right = int(box[2]), int(box[3])
        image = cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

    cv2.imshow("test", image)

    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()