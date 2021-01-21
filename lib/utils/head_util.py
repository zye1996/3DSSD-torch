import tensorflow as tf
import numpy as np
# import utils.tf_util as tf_util
import lib.dataset.maps_dict as maps_dict
import torch
import torch.nn as nn
import lib.pointnet2.pytorch_utils as pt_utils

from functools import partial
from lib.core.config import cfg


def box_regression_head_tf(feature_input, pred_cls_channel, pred_reg_base_num, pred_reg_channel_num, bn, is_training, pred_attr_velo, conv_op, bn_decay, output_dict):
    """
    Construct box-regression head
    """
    bs, points_num, _ = feature_input.get_shape().as_list()
    # classification
    pred_cls = conv_op(feature_input, 128, scope='pred_cls_base', bn=bn, is_training=is_training, bn_decay=bn_decay)
    pred_cls = conv_op(pred_cls, pred_cls_channel, activation_fn=None, scope='pred_cls')

    # bounding-box prediction 
    pred_reg = conv_op(feature_input, 128, bn=bn, is_training=is_training, scope='pred_reg_base', bn_decay=bn_decay)
    pred_reg = conv_op(pred_reg, pred_reg_base_num * (pred_reg_channel_num + cfg.MODEL.ANGLE_CLS_NUM * 2), activation_fn=None, scope='pred_reg')
    pred_reg = tf.reshape(pred_reg, [bs, points_num, pred_reg_base_num, pred_reg_channel_num + cfg.MODEL.ANGLE_CLS_NUM * 2])

    if pred_attr_velo: # velocity and attribute
        pred_attr = conv_op(feature_input, 128, bn=bn, is_training=is_training, scope='pred_attr_base', bn_decay=bn_decay)
        pred_attr = conv_op(pred_attr, pred_reg_base_num * 8, activation_fn=None, scope='pred_attr')
        pred_attr = tf.reshape(pred_attr, [bs, points_num, pred_reg_base_num, 8])

        pred_velo = conv_op(feature_input, 128, bn=bn, is_training=is_training, scope='pred_velo_base', bn_decay=bn_decay)
        pred_velo = conv_op(pred_velo, pred_reg_base_num * 2, activation_fn=None, scope='pred_velo')
        pred_velo = tf.reshape(pred_velo, [bs, points_num, pred_reg_base_num, 2])

        output_dict[maps_dict.PRED_ATTRIBUTE].append(pred_attr)
        output_dict[maps_dict.PRED_VELOCITY].append(pred_velo)

    output_dict[maps_dict.PRED_CLS].append(pred_cls)
    output_dict[maps_dict.PRED_OFFSET].append(tf.slice(pred_reg, [0, 0, 0, 0], [-1, -1, -1, pred_reg_channel_num]))
    output_dict[maps_dict.PRED_ANGLE_CLS].append(tf.slice(pred_reg, \
         [0, 0, 0, pred_reg_channel_num], [-1, -1, -1, cfg.MODEL.ANGLE_CLS_NUM]))
    output_dict[maps_dict.PRED_ANGLE_RES].append(tf.slice(pred_reg, \
         [0, 0, 0, pred_reg_channel_num+cfg.MODEL.ANGLE_CLS_NUM], [-1, -1, -1, -1]))

    return
    
    
def iou_regression_head(feature_input, pred_cls_channel, bn, is_training, conv_op, bn_decay, output_dict):
    """
    Construct iou-prediction head:
    """
    bs, points_num, _ = feature_input.get_shape().as_list()
    # classification
    pred_iou = conv_op(feature_input, 128, scope='pred_iou_base', bn=bn, is_training=is_training, bn_decay=bn_decay)
    pred_iou = conv_op(pred_iou, pred_cls_channel, activation_fn=None, scope='pred_iou')

    output_dict[maps_dict.PRED_IOU_3D_VALUE].append(pred_iou)

    return


class BoxRegressionHead(nn.Module):
    def __init__(self, pred_cls_channel, pred_reg_base_num, pred_reg_channel_num, is_training, pred_attr_velo, bn, pre_channel):
        super().__init__()
        self.pred_cls_channel = pred_cls_channel
        self.pred_reg_base_num = pred_reg_base_num
        self.pred_reg_channel_num = pred_reg_channel_num
        self.is_training = is_training
        self.pred_attr_velo = pred_attr_velo
        self.pre_channel = pre_channel

        self.bn = bn

        cls_layers = []
        cls_layers.append(pt_utils.Conv1d(self.pre_channel, 128, bn=self.bn))
        cls_layers.append(pt_utils.Conv1d(128, self.pred_cls_channel, bn=False, activation=None))
        self.cls_layers = nn.Sequential(*cls_layers)

        self.angle_cls_num = cfg.MODEL.ANGLE_CLS_NUM
        reg_layers = []
        reg_layers.append(pt_utils.Conv1d(pre_channel, 128, bn=self.bn))
        reg_channel = pred_reg_base_num * (pred_reg_channel_num + self.angle_cls_num * 2)
        reg_layers.append(pt_utils.Conv1d(128, reg_channel, bn=False, activation=None))
        self.reg_layers = nn.Sequential(*reg_layers)

    def forward(self, feature_input, output_dict):
        bs, points_num = feature_input.shape[0], feature_input.shape[1]

        feature_input_transpose = feature_input.transpose(1, 2)
        pred_cls = self.cls_layers(feature_input_transpose).transpose(1, 2)
        pred_reg = self.reg_layers(feature_input_transpose).transpose(1, 2)
        pred_reg = pred_reg.view(bs, points_num, self.pred_reg_base_num, self.pred_reg_channel_num + self.angle_cls_num * 2)

        output_dict[maps_dict.PRED_CLS].append(pred_cls)
        output_dict[maps_dict.PRED_OFFSET].append(pred_reg[:, :, :, 0:self.pred_reg_channel_num])
        output_dict[maps_dict.PRED_ANGLE_CLS].append(pred_reg[:, :, :, self.pred_reg_channel_num:self.pred_reg_channel_num+self.angle_cls_num])
        output_dict[maps_dict.PRED_ANGLE_RES].append(pred_reg[:, :, :, self.pred_reg_channel_num+self.angle_cls_num:])
        return output_dict



