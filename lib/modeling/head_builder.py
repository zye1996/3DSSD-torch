import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch

from lib.core.config import cfg
from lib.utils.head_util import BoxRegressionHead
import lib.pointnet2.pytorch_utils as pt_utils


import lib.dataset.maps_dict as maps_dict


class HeadBuilder(nn.Module):
    def __init__(self, batch_size, anchor_num, head_idx, head_cfg, is_training):
        super().__init__()
        self.is_training = is_training
        self.head_idx = head_idx
        self.anchor_num = anchor_num
        self.batch_size = batch_size 

        cur_head = head_cfg

        self.xyz_index = cur_head[0]
        self.feature_index = cur_head[1]
        self.op_type = cur_head[2]
        self.mlp_list = cur_head[3]
        self.bn = cur_head[4]
        self.layer_type = cur_head[5]
        self.scope = cur_head[6]

        if head_idx == 0: # stage 1
            self.head_cfg = cfg.MODEL.FIRST_STAGE
        elif head_idx == 1: # stage 2
            self.head_cfg = cfg.MODEL.SECOND_STAGE
        else: raise Exception('Not Implementation Error!!!') # stage 3

        # determine channel number
        if self.head_cfg.CLS_ACTIVATION == 'Sigmoid':
            self.pred_cls_channel = self.anchor_num
        elif self.head_cfg.CLS_ACTIVATION == 'Softmax':
            self.pred_cls_channel = self.anchor_num + 1
        if self.layer_type == 'IoU':
            self.pred_cls_channel = self.anchor_num

        self.reg_method = self.head_cfg.REGRESSION_METHOD.TYPE 
        anchor_type = self.reg_method.split('-')[-1] # Anchor & free

        pred_reg_base_num = {
            'Anchor': self.anchor_num,
            'free': 1,
        } 
        self.pred_reg_base_num = pred_reg_base_num[anchor_type]

        pred_reg_channel_num = {
            'Dist-Anchor': 6,
            'Log-Anchor': 6,
            'Dist-Anchor-free': 6,
            # bin_x/res_x/bin_z/res_z/res_y/res_size
            'Bin-Anchor': self.head_cfg.REGRESSION_METHOD.BIN_CLASS_NUM * 4 + 4,
        } 
        self.pred_reg_channel_num = pred_reg_channel_num[self.reg_method]

        self.pre_channel = cfg.MODEL.NETWORK.FIRST_STAGE.ARCHITECTURE[self.feature_index[0]-1][15]
        layer_modules = []
        pre_channel = self.pre_channel
        for i in range(len(self.mlp_list)):
            layer_modules.append(pt_utils.Conv1d(pre_channel, self.mlp_list[i], bn=self.bn))
            pre_channel = self.mlp_list[i]
        self.layer_modules = nn.Sequential(*layer_modules)

        self.head_predictor = BoxRegressionHead(self.pred_cls_channel, self.pred_reg_base_num, self.pred_reg_channel_num,
                                                self.is_training, self.head_cfg.PREDICT_ATTRIBUTE_AND_VELOCITY,
                                                self.bn, pre_channel)

    def forward(self, xyz_list, feature_list, output_dict):
        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])
        xyz_input = torch.cat(xyz_input, dim=1)  # bs, npoint, 3
        
        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])
        feature_input = torch.cat(feature_input, dim=1) # bs, npoint, c

        feature_input_transpose = feature_input.transpose(1, 2)
        feature_input = self.layer_modules(feature_input_transpose).transpose(1, 2)
        output_dict = self.head_predictor(feature_input, output_dict)

        if self.layer_type == 'Det': # only add xyz and feature in 'Det' mode
            output_dict[maps_dict.KEY_OUTPUT_XYZ].append(xyz_input)
            output_dict[maps_dict.KEY_OUTPUT_FEATURE].append(feature_input)

        return xyz_input, feature_input, output_dict
