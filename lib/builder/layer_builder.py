import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch

from lib.utils.layers_util import Pointnet_sa_module_msg, Vote_layer

import lib.dataset.maps_dict as maps_dict


class LayerBuilder(nn.Module):
    def __init__(self, layer_idx, is_training, layer_cfg, fps_idx_list=None, bn_decay=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_training = is_training

        self.layer_architecture = layer_cfg[self.layer_idx]

        self.xyz_index = self.layer_architecture[0]
        self.feature_index = self.layer_architecture[1]
        self.radius_list = self.layer_architecture[2]
        self.nsample_list = self.layer_architecture[3]
        self.mlp_list = self.layer_architecture[4]
        self.bn = self.layer_architecture[5]

        self.fps_sample_range_list = self.layer_architecture[6]
        self.fps_method_list = self.layer_architecture[7]
        self.npoint_list = self.layer_architecture[8]
        assert len(self.fps_sample_range_list) == len(self.fps_method_list)
        assert len(self.fps_method_list) == len(self.npoint_list)

        self.former_fps_idx = self.layer_architecture[9]
        self.use_attention = self.layer_architecture[10]
        self.layer_type = self.layer_architecture[11]
        self.scope = self.layer_architecture[12] 
        self.dilated_group = self.layer_architecture[13]
        self.vote_ctr_index = self.layer_architecture[14]
        self.aggregation_channel = self.layer_architecture[15]

        if self.layer_type in ['SA_Layer', 'Vote_Layer', 'SA_Layer_SSG_Last']:
            assert len(self.xyz_index) == 1
        elif self.layer_type == 'FP_Layer':
            assert len(self.xyz_index) == 2
        else: raise Exception('Not Implementation Error!!!')

        if layer_idx == 0:
            self.pre_channel = 1
        else:
            self.pre_channel = layer_cfg[self.feature_index[0]-1][15]

        if self.layer_type == 'SA_Layer':
            self.layer_module = Pointnet_sa_module_msg(
                self.radius_list, self.nsample_list,
                self.mlp_list, self.is_training, bn_decay, self.bn,
                self.fps_sample_range_list, self.fps_method_list, self.npoint_list,
                self.use_attention, self.scope,
                self.dilated_group, self.aggregation_channel, self.pre_channel)
        elif self.layer_type == 'Vote_Layer':
            self.layer_module = Vote_layer(self.mlp_list, self.bn, self.is_training, self.pre_channel)


    def forward(self, xyz_list, feature_list, fps_idx_list, output_dict):
        """
        Build layers
        """

        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])
 
        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])

        if self.former_fps_idx != -1:
            former_fps_idx = fps_idx_list[self.former_fps_idx]
        else:
            former_fps_idx = None

        if self.vote_ctr_index != -1:
            vote_ctr = xyz_list[self.vote_ctr_index]
        else:
            vote_ctr = None

        if self.layer_type == 'SA_Layer':
            new_xyz, new_points, new_fps_idx = self.layer_module(xyz_input[0], feature_input[0], former_fps_idx, vote_ctr)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(new_fps_idx)

        elif self.layer_type == 'Vote_Layer':
            new_xyz, new_points, ctr_offsets = self.layer_module(xyz_input[0], feature_input[0])
            output_dict[maps_dict.PRED_VOTE_BASE].append(xyz_input[0])
            output_dict[maps_dict.PRED_VOTE_OFFSET].append(ctr_offsets)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(None)

        return xyz_list, feature_list, fps_idx_list, output_dict


