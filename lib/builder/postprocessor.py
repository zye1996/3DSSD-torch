import numpy as np
# import tensorflow as tf
import torch
import torchvision

from lib.core.config import cfg
from lib.utils.anchors_util import project_to_bev_torch
from lib.utils.box_3d_utils import box_3d_to_anchor_torch

import lib.dataset.maps_dict as maps_dict


class PostProcessor:
    def __init__(self, stage, cls_num):
        if stage == 0:
            self.postprocessor_cfg = cfg.MODEL.FIRST_STAGE
        elif stage == 1:
            self.postprocessor_cfg = cfg.MODEL.SECOND_STAGE
        else: raise Exception('Not Implementation Error')

        self.max_output_size = self.postprocessor_cfg.MAX_OUTPUT_NUM
        self.nms_threshold = self.postprocessor_cfg.NMS_THRESH

        self.cls_num = cls_num
   
    
    def class_unaware_format(self, pred_anchors_3d, pred_score):
        """ (for rpn propose)
        Change prediction format from class-aware-format to class-ignorance-format
        pred_anchors_3d: [bs, points_num, 1/cls_num, 7]
        pred_score: [bs, points_num, cls_num]

        return: pred_anchors_3d: [bs, points_num, 1, 7]
                pred_score: [bs, points_num, 1]
        """ 
        unaware_pred_score = torch.max(pred_score, dim=-1, keepdim=True)[0]
        cls_num = pred_anchors_3d.shape[2]
        if cls_num == 1:
            return pred_anchors_3d, unaware_pred_score

        # class-aware in boundingbox prediction
        pred_cls = torch.max(pred_score, dim=-1)[1]
        pred_cls_onehot = tf.cast(tf.one_hot(pred_cls, depth=cls_num, on_value=1, off_value=0, axis=-1), tf.float32)
        # bs, pts_num, cls_num, 7
        unaware_pred_anchors_3d = pred_anchors_3d * tf.expand_dims(pred_cls_onehot, axis=-1)
        unaware_pred_anchors_3d = tf.reduce_sum(unaware_pred_anchors_3d, axis=2, keepdims=True)
        return unaware_pred_anchors_3d, unaware_pred_score


    def forward(self, pred_anchors_3d, pred_score, output_dict, pred_attribute=None, pred_velocity=None):
        """
        pred_anchors_3d: [bs, points_num, 1/cls_num, 7]
        pred_score: [bs, points_num, cls_num]
        pred_attribute: [bs, points_num, 1/cls_num, 8]
        pred_velocity: [bs, points_num, 1/cls_num, 2]
        """
        cls_num = pred_score.shape[-1]
        if cls_num != self.cls_num: # format predictions to class-unaware predictions
            assert pred_attribute == None and pred_velocity == None, 'Not support the predictions of attribute and velocity in RPN phase'
            pred_anchors_3d, pred_score = self.class_unaware_format(pred_anchors_3d, pred_score)

        pred_anchors_3d_list = torch.split(pred_anchors_3d, 1, dim=0)
        pred_scores_list = torch.split(pred_score, 1, dim=0)

        pred_3d_bbox_list = []
        pred_3d_cls_score_list = []
        pred_3d_cls_cat_list = []
        pred_attribute_list = []
        pred_velocity_list = []
        for batch_idx, pred_anchors_3d, pred_scores in zip(range(len(pred_anchors_3d_list)), pred_anchors_3d_list, pred_scores_list):
            cur_pred_3d_bbox_list = []
            cur_pred_3d_cls_score_list = []
            cur_pred_3d_cls_cat_list = []
            cur_pred_attribute_list = []
            cur_pred_velocity_list = []

            pred_anchors_3d = pred_anchors_3d.squeeze(0)
            pred_scores = pred_scores.squeeze(0)

            for i in range(self.cls_num):
                reg_i = min(i, pred_anchors_3d.shape[1] - 1)
                cur_pred_anchors_3d = pred_anchors_3d[:, reg_i, :] 

                cur_pred_anchors = box_3d_to_anchor_torch(cur_pred_anchors_3d)
                cur_pred_anchors_bev = project_to_bev_torch(cur_pred_anchors) # [-1, 4]

                cur_cls_score = pred_scores[:, i]
                nms_index = torchvision.ops.nms(cur_pred_anchors_bev, cur_cls_score, self.nms_threshold)
               
                cur_pred_3d_bbox_list.append(cur_pred_anchors_3d[nms_index])
                cur_pred_3d_cls_score_list.append(cur_cls_score[nms_index])
                cur_pred_3d_cls_cat_list.append(torch.ones_like(nms_index).long().to(nms_index.device))

                if pred_attribute is not None:
                    cur_pred_attribute_list.append(pred_attribute[batch_idx, :, reg_i, :][nms_index])
                if pred_velocity is not None:
                    cur_pred_velocity_list.append(pred_velocity[batch_idx, :, reg_i, :][nms_index])

            cur_pred_3d_bbox_list = torch.cat(cur_pred_3d_bbox_list, dim=0)
            cur_pred_3d_cls_score_list = torch.cat(cur_pred_3d_cls_score_list, dim=0)
            cur_pred_3d_cls_cat_list = torch.cat(cur_pred_3d_cls_cat_list, dim=0)

            pred_3d_bbox_list.append(cur_pred_3d_bbox_list)
            pred_3d_cls_score_list.append(cur_pred_3d_cls_score_list)
            pred_3d_cls_cat_list.append(cur_pred_3d_cls_cat_list)

            if pred_attribute is not None:
                cur_pred_attribute_list = torch.cat(cur_pred_attribute_list, dim=0)
                pred_attribute_list.append(cur_pred_attribute_list)

            if pred_velocity is not None:
                cur_pred_velocity_list = torch.cat(cur_pred_velocity_list, dim=0)
                pred_velocity_list.append(cur_pred_velocity_list)

        # pred_3d_bbox_list = torch.stack(pred_3d_bbox_list, dim=0)
        # pred_3d_cls_score_list = torch.stack(pred_3d_cls_score_list, dim=0)
        # pred_3d_cls_cat_list = torch.stack(pred_3d_cls_cat_list, dim=0)
            
        output_dict[maps_dict.PRED_3D_BBOX] = pred_3d_bbox_list
        output_dict[maps_dict.PRED_3D_SCORE] = pred_3d_cls_score_list
        output_dict[maps_dict.PRED_3D_CLS_CATEGORY] = pred_3d_cls_cat_list
        if pred_attribute is not None:
            output_dict[maps_dict.PRED_3D_ATTRIBUTE] = pred_attribute_list
        if pred_velocity is not None:
            output_dict[maps_dict.PRED_3D_VELOCITY] = pred_velocity_list

        return output_dict
