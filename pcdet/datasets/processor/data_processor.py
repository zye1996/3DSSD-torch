from functools import partial
from easydict import EasyDict as edict

import numpy as np

from ...utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        data_dict['choice'] = choice
        return data_dict

    def sample_points_voxel(self, data_dict=None, config=None, voxel_generator=None):

        # sample from cur_frame
        cur_frame_points = config.NUM_POINTS
        data_dict = self.transform_points_to_voxels(data_dict, config, voxel_generator=voxel_generator)

        voxel_mask = data_dict['voxel_num_points'] > 0
        voxel = data_dict['voxels'][voxel_mask]
        voxel_num_points = data_dict['voxel_num_points'][voxel_mask]
        voxel_coords = data_dict['voxel_coords'][voxel_mask]

        data_dict = {'points': voxel_coords}
        sample_config = edict({'NUM_POINTS': {self.mode: cur_frame_points}})
        data_dict = self.sample_points(data_dict=data_dict, config=sample_config)
        voxel_mask = data_dict['choice']

        voxel = voxel[voxel_mask]
        voxel_num_points = voxel_num_points[voxel_mask]

        idx = np.random.sample(voxel.shape[0])
        idx = np.floor(idx * voxel_num_points)
        cur_sample = np.take(voxel, idx, axis=1)

        data_dict['points'] = cur_sample
        return data_dict

    def sample_points_sweep(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.sample_points, config=config)

        nums_points = config.NUM_POINTS[self.mode]
        if not isinstance(nums_points, list):
            return data_dict

        points = data_dict['points']
        time = points[-1, :]

        cur_frame_points = num_points[0]
        prev_frame_points = num_points[1]

        voxel_generator = VoxelGenerator(
            voxel_size=config.VOXEL_SIZE,
            point_cloud_range=self.point_cloud_range,
            max_num_points=config.MAX_POINTS_PER_VOXEL,
            max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
        )

        # sample from cur_frame
        data_dict_temp = {'points': points[time == time[0]]}
        sample_config = edict({'NUM_POINTS': cur_frame_points})
        cur_sample = self.sample_points_voxel(data_dict_temp, config=sample_config, voxel_generator=voxel_generator)['points']

        # sample from previous
        data_dict_temp = edict({'points': points[time != time[0]]})
        sample_config = edict({'NUM_POINTS': prev_frame_points})
        prev_sample = self.sample_points_voxel(data_dict_temp, config=sample_config, voxel_generator=voxel_generator)['points']

        data_dict['points'] = np.concatenate([cur_sample, prev_sample], axis=0)

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
