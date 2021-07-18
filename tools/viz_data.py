import argparse
import glob
import os
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import pandas as pd
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, pred_path=None, logger=None, ext='.bin'):
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
        self.pred_path = pred_path
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
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class DemoDatasetNuscenes(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, pred_path=None, logger=None, ext='.bin'):
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
        self.pred_path = pred_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def _parse_pred_file(self, file):

        preds = {}
        
        df = pd.read_csv(file, delimiter=' ', header=None, index_col=None)
        preds['pred_boxes'] = torch.tensor(df.iloc[:, 3:10].values)
        preds['pred_scores'] = torch.tensor(df.iloc[:, 2].values)
        preds['pred_labels'] = torch.tensor(df.iloc[:, 1].values)

        return preds

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        basename = os.path.splitext(os.path.basename(self.sample_file_list[index]))[0]
        if self.pred_path is not None:
            pred_file = Path(self.pred_path / (basename + '.txt'))
        else:
            pred_file = Path(self.dataset_cfg.DATA_PATH) / 'second_pred' / (basename + '.txt')

        preds = {"pred_boxes": [],
                 "pred_scores": [],
                 "pred_labels": []}
        if os.path.exists(pred_file):
            preds = self._parse_pred_file(pred_file)

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        input_dict.update(preds)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--pred_path', type=str, default='second_pred')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDatasetNuscenes(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            V.draw_scenes(points=data_dict['points'][:, 1:], ref_boxes=data_dict['pred_boxes'][0],
                          ref_scores=data_dict['pred_scores'][0], ref_labels=data_dict['pred_labels'][0])
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
