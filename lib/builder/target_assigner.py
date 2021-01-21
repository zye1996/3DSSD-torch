import numpy as np
import tensorflow as tf

import torch

from lib.core.config import cfg

from lib.utils.voxelnet_aug import check_inside_points
from lib.pointnet2.pointnet2_utils import grouping_operation

# from utils.tf_ops.grouping.tf_grouping import group_point, query_points_iou
# from utils.tf_ops.evaluation.tf_evaluate import calc_iou
# import np_functions.gt_sampler as gt_sampler

class TargetAssigner:
    def __init__(self, stage):
        """
        stage: TargetAssigner of stage1 or stage2
        """
        if stage == 0:
            cur_cfg_file = cfg.MODEL.FIRST_STAGE 
        elif stage == 1:
            cur_cfg_file = cfg.MODEL.SECOND_STAGE 
        else:
            raise Exception('Not Implementation Error')

        self.assign_method = cur_cfg_file.ASSIGN_METHOD
        self.iou_sample_type = cur_cfg_file.IOU_SAMPLE_TYPE

        # some parameters
        self.minibatch_size = cur_cfg_file.MINIBATCH_NUM
        self.positive_ratio = cur_cfg_file.MINIBATCH_RATIO 
        self.pos_iou = cur_cfg_file.CLASSIFICATION_POS_IOU
        self.neg_iou = cur_cfg_file.CLASSIFICATION_NEG_IOU
        self.effective_sample_range = cur_cfg_file.CLASSIFICATION_LOSS.SOFTMAX_SAMPLE_RANGE


        if self.assign_method == 'IoU':
            self.assign_targets_anchors = self.iou_assign_targets_anchors
        elif self.assign_method == 'Mask':
            self.assign_targets_anchors = self.mask_assign_targets_anchors

    def assign(self, points, anchors_3d, gt_boxes_3d, gt_labels, gt_angle_cls, gt_angle_res, gt_velocity=None, gt_attribute=None, valid_mask=None):
        """
        points: [bs, points_num, 3]
        anchors_3d: [bs, points_num, cls_num, 7]
        gt_boxes_3d: [bs, gt_num, 7]
        gt_labels: [bs, gt_num]
        gt_angle_cls: [bs, gt_num]
        gt_angle_res: [bs, gt_num]
        gt_velocity: [bs, gt_num, 2]
        gt_attribute: [bs, gt_num]

        return: [bs, points_num, cls_num]
        """
        bs, points_num, cls_num = anchors_3d.shape[0], anchors_3d.shape[1], anchors_3d.shape[2]

        if valid_mask is None:
            valid_mask = torch.ones([bs, points_num, cls_num]).float().to(points.device)

        assigned_idx, assigned_pmask, assigned_nmask = self.assign_targets_anchors(points, anchors_3d, gt_boxes_3d, gt_labels, valid_mask) # [bs, points_num, cls_num] 

        assigned_gt_labels = torch.gather(gt_labels.unsqueeze(dim=-1), 1, assigned_idx)
        # assigned_gt_labels = self.gather_class(gt_labels, assigned_idx) # [bs, points_num, cls_num]
        assigned_gt_labels = assigned_gt_labels * assigned_pmask.long()
        assigned_gt_labels = torch.sum(assigned_gt_labels, dim=-1)

        # assigned_gt_boxes_3d = group_point(gt_boxes_3d, assigned_idx)
        assigned_gt_boxes_3d = torch.gather(gt_boxes_3d, 1, assigned_idx.repeat((1, 1, gt_boxes_3d.shape[2]))).unsqueeze(dim=2)
        assigned_gt_angle_cls = torch.gather(gt_angle_cls.unsqueeze(dim=-1), 1, assigned_idx)
        assigned_gt_angle_res = torch.gather(gt_angle_res.unsqueeze(dim=-1), 1, assigned_idx)

        if gt_velocity is not None:
            # bs, npoint, cls_num, 2 
            assigned_gt_velocity = group_point(gt_velocity, assigned_idx)
        else:
            assigned_gt_velocity = None

        if gt_attribute is not None:
            # bs, npoint, cls_num
            assigned_gt_attribute = self.gather_class(gt_attribute, assigned_idx)
        else:
            assigned_gt_attribute = None

        returned_list = [assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute]

        return returned_list



    def gather_class(self, gt_labels, assigned_idx):
        # [bs, gt_num] -> [bs, points_num, cls_num]
        # gt_labels_dtype = gt_labels.dtype
        gt_labels_f = gt_labels.unsqueeze(dim=1).float()
        assigned_gt_labels = grouping_operation(gt_labels_f, assigned_idx.int())
        assigned_gt_labels = assigned_gt_labels.squeeze(dim=-1).long().transpose(1, 2)
        return assigned_gt_labels
        

    def iou_assign_targets_anchors(self, points, anchors_3d, gt_boxes_3d, gt_labels, valid_mask):
        """
        Assign targets for each anchor
        points: [bs, points_num, 3]
        anchors_3d: [bs, points_num, cls_num, 7]
        gt_boxes_3d: [bs, gt_boxes_3d, 7]
        gt_labels: [bs, gt_boxes_3d]
        valid_mask: [bs, points_num, cls_num]

        Return:
        assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
        assigned_pmask: [bs, points_num, cls_num]
        assigned_nmask: [bs, points_num, cls_num]
        """
        # first calculate IoU
        bs, points_num, cls_num, _ = anchors_3d.get_shape().as_list()
        gt_num = tf.shape(gt_boxes_3d)[1]
        anchors_3d_reshape = tf.reshape(anchors_3d, [bs, points_num * cls_num, 7])

        # bs, pts_num * cls_num, gt_num
        iou_bev, iou_3d = calc_iou(anchors_3d_reshape, gt_boxes_3d)
        if self.iou_sample_type == 'BEV':
            iou_matrix = iou_bev
        elif self.iou_sample_type == '3D':
            iou_matrix = iou_3d 
        elif self.iou_sample_type == 'Point': # point_iou
            iou_matrix = query_points_iou(points, anchors_3d_reshape, gt_boxes_3d, iou_3d)
        iou_matrix = tf.reshape(iou_matrix, [bs, points_num, cls_num, gt_num])
        
        assigned_idx, assigned_pmask, assigned_nmask = tf.py_func(gt_sampler.iou_assign_targets_anchors_np, 
            [iou_matrix, points, anchors_3d, gt_boxes_3d, gt_labels, self.minibatch_size, self.positive_ratio, self.pos_iou, self.neg_iou, self.effective_sample_range, valid_mask], 
            [tf.int32, tf.float32, tf.float32])
 
        assigned_idx = tf.reshape(assigned_idx, [bs, points_num, cls_num])
        assigned_pmask = tf.reshape(assigned_pmask, [bs, points_num, cls_num])
        assigned_nmask = tf.reshape(assigned_nmask, [bs, points_num, cls_num])
        return assigned_idx, assigned_pmask, assigned_nmask


    def mask_assign_targets_anchors(self, points, anchors_3d, gt_boxes_3d, gt_labels, valid_mask):
        """
        Assign targets for each anchor
        points: [bs, points_num, 3]
        anchors_3d: [bs, points_num, cls_num, 3] centers of anchors
        gt_boxes_3d: [bs, gt_boxes_3d, 7]
        gt_labels: [bs, gt_boxes_3d]
        valid_mask: [bs, points_num, cls_num]

        Return:
        assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
        assigned_pmask: [bs, points_num, cls_num]
        assigned_nmask: [bs, points_num, cls_num]
        """
        bs, points_num, cls_num = anchors_3d.shape[0], anchors_3d.shape[1], anchors_3d.shape[2]
        gt_num = gt_boxes_3d.shape[1]

        # then let's calculate whether a point is within a gt_boxes_3d
        assigned_idx, assigned_pmask, assigned_nmask = self.__mask_assign_targets_anchors_torch(points, anchors_3d,
                                                                                                gt_boxes_3d,
                                                                                                gt_labels,
                                                                                                self.minibatch_size,
                                                                                                self.positive_ratio,
                                                                                                self.pos_iou,
                                                                                                self.neg_iou,
                                                                                                self.effective_sample_range,
                                                                                                valid_mask)

        assigned_idx = assigned_idx.view(bs, points_num, cls_num)
        assigned_pmask = assigned_pmask.view(bs, points_num, cls_num)
        assigned_nmask = assigned_nmask.view(bs, points_num, cls_num)
        return assigned_idx, assigned_pmask, assigned_nmask

    def __mask_assign_targets_anchors_torch(self, batch_points, batch_anchors_3d, batch_gt_boxes_3d,
                                            batch_gt_labels,
                                            minibatch_size, positive_rate, pos_iou, neg_iou, effective_sample_range,
                                            valid_mask):
        """ Mask assign targets function
        batch_points: [bs, points_num, 3]
        batch_anchors_3d: [bs, points_num, cls_num, 7]
        batch_gt_boxes_3d: [bs, gt_num, 7]
        batch_gt_labels: [bs, gt_num]
        valid_mask: [bs, points_num, cls_num]

        return:
            assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
            assigned_pmask: [bs, points_num, cls_num], float32
            assigned_nmask: [bs, points_num, cls_num], float32
        """
        bs, pts_num, cls_num, _ = batch_anchors_3d.shape

        positive_size = int(minibatch_size * positive_rate)

        batch_assigned_idx = torch.zeros([bs, pts_num, cls_num]).long().to(batch_points.device)
        batch_assigned_pmask = torch.zeros([bs, pts_num, cls_num]).float().to(batch_points.device)
        batch_assigned_nmask = torch.zeros([bs, pts_num, cls_num]).float().to(batch_points.device)

        for i in range(bs):
            cur_points = batch_points[i]
            cur_anchors_3d = batch_anchors_3d[i]  # [pts_num, cls_num, 3/7]
            cur_valid_mask = valid_mask[i]  # [pts_num, cls_num]

            # gt_num
            cur_gt_labels = batch_gt_labels[i]  # [gt_num]
            cur_gt_boxes_3d = batch_gt_boxes_3d[i]  # [gt_num, 7]

            # first filter gt_boxes
            filter_idx = torch.where(torch.any(torch.not_equal(cur_gt_boxes_3d, 0), axis=-1))[0].to(cur_gt_labels.device)
            cur_gt_labels = cur_gt_labels[filter_idx]
            cur_gt_boxes_3d = cur_gt_boxes_3d[filter_idx]

            cur_points_numpy = cur_points.cpu().detach().numpy()
            cur_gt_boxes_3d_numpy = cur_gt_boxes_3d.cpu().detach().numpy()

            points_mask_numpy = check_inside_points(cur_points_numpy, cur_gt_boxes_3d_numpy)  # [pts_num, gt_num]
            points_mask = torch.from_numpy(points_mask_numpy).int().to(cur_points.device)
            sampled_gt_idx_numpy = np.argmax(points_mask_numpy, axis=-1)
            sampled_gt_idx = torch.from_numpy(sampled_gt_idx_numpy).long().to(cur_points.device)  # [pts_num]
            # used for label_mask
            assigned_gt_label = cur_gt_labels[sampled_gt_idx]  # [pts_num]
            assigned_gt_label = assigned_gt_label - 1  # 1... -> 0...
            # used for dist_mask
            assigned_gt_boxes = cur_gt_boxes_3d[sampled_gt_idx]  # [pts_num, 7]
            # then calc the distance between anchors and assigned_boxes
            dist = cur_anchors_3d[:, :, :3] - assigned_gt_boxes[:, 0:3].unsqueeze(dim=1).repeat((1, cur_anchors_3d.shape[1], 1))
            dist = torch.sqrt(torch.sum(dist * dist, dim=-1))

            # dist = np.linalg.norm(cur_anchors_3d[:, :, :3] - assigned_gt_boxes[:, np.newaxis, :3],
            #                       axis=-1)

            filtered_assigned_idx = filter_idx[sampled_gt_idx]  # [pts_num]
            filtered_assigned_idx = filtered_assigned_idx.view(pts_num, 1).repeat((1, cls_num))
            batch_assigned_idx[i] = filtered_assigned_idx

            # then we generate pos/neg mask
            if cls_num == 1:  # anchor_free
                label_mask = torch.ones((pts_num, cls_num)).float().to(points_mask.device)
            else:  # multiple anchors
                label_mask = np.tile(np.reshape(np.arange(cls_num), [1, cls_num]), [pts_num, 1])
                label_mask = np.equal(label_mask, assigned_gt_label[:, np.newaxis]).astype(np.float32)

            pmask = torch.max(points_mask, dim=1)[0] > 0
            dist_mask = torch.less_equal(dist, effective_sample_range)  # pts_num, cls_num
            pmask = torch.logical_and(pmask.unsqueeze(-1), dist_mask)
            pmask = pmask.float() * label_mask
            pmask = pmask * cur_valid_mask

            nmask = torch.max(points_mask, dim=1)[0] == 0
            nmask = nmask.view(pts_num, 1).repeat((1, cls_num))
            nmask = nmask.float() * label_mask
            nmask = nmask * cur_valid_mask


            # then randomly sample
            if minibatch_size != -1:
                pts_pmask = np.any(pmask, axis=1)  # pts_num
                pts_nmask = np.any(nmask, axis=1)  # [pts_num]

                positive_inds = np.where(pts_pmask)[0]
                cur_positive_num = np.minimum(len(positive_inds), positive_size)
                if cur_positive_num > 0:
                    positive_inds = np.random.choice(positive_inds, cur_positive_num, replace=False)
                pts_pmask = np.zeros_like(pts_pmask)
                pts_pmask[positive_inds] = 1

                cur_negative_num = minibatch_size - cur_positive_num
                negative_inds = np.where(pts_nmask)[0]
                cur_negative_num = np.minimum(len(negative_inds), cur_negative_num)
                if cur_negative_num > 0:
                    negative_inds = np.random.choice(negative_inds, cur_negative_num, replace=False)
                pts_nmask = np.zeros_like(pts_nmask)
                pts_nmask[negative_inds] = 1

                pmask = pmask * pts_pmask[:, np.newaxis]
                nmask = nmask * pts_nmask[:, np.newaxis]

            batch_assigned_pmask[i] = pmask
            batch_assigned_nmask[i] = nmask
        return batch_assigned_idx, batch_assigned_pmask, batch_assigned_nmask


    def __mask_assign_targets_anchors_np(self, batch_points, batch_anchors_3d, batch_gt_boxes_3d,
                                         batch_gt_labels,
                                         minibatch_size, positive_rate, pos_iou, neg_iou,
                                         effective_sample_range, valid_mask):
        """ Mask assign targets function
        batch_points: [bs, points_num, 3]
        batch_anchors_3d: [bs, points_num, cls_num, 7]
        batch_gt_boxes_3d: [bs, gt_num, 7]
        batch_gt_labels: [bs, gt_num]
        valid_mask: [bs, points_num, cls_num]
        return:
            assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
            assigned_pmask: [bs, points_num, cls_num], float32
            assigned_nmask: [bs, points_num, cls_num], float32
        """
        bs, pts_num, cls_num, _ = batch_anchors_3d.shape

        positive_size = int(minibatch_size * positive_rate)

        batch_assigned_idx = np.zeros([bs, pts_num, cls_num], np.int32)
        batch_assigned_pmask = np.zeros([bs, pts_num, cls_num], np.float32)
        batch_assigned_nmask = np.zeros([bs, pts_num, cls_num], np.float32)

        for i in range(bs):
            cur_points = batch_points[i]
            cur_anchors_3d = batch_anchors_3d[i] # [pts_num, cls_num, 3/7]
            cur_valid_mask = valid_mask[i] # [pts_num, cls_num]

            # gt_num
            cur_gt_labels = batch_gt_labels[i] # [gt_num]
            cur_gt_boxes_3d = batch_gt_boxes_3d[i] # [gt_num, 7]

            # first filter gt_boxes
            filter_idx = np.where(np.any(np.not_equal(cur_gt_boxes_3d, 0), axis=-1))[0]
            cur_gt_labels = cur_gt_labels[filter_idx]
            cur_gt_boxes_3d = cur_gt_boxes_3d[filter_idx]

            points_mask = check_inside_points(cur_points, cur_gt_boxes_3d) # [pts_num, gt_num]
            sampled_gt_idx = np.argmax(points_mask, axis=-1) # [pts_num]
            # used for label_mask
            assigned_gt_label = cur_gt_labels[sampled_gt_idx] # [pts_num]
            assigned_gt_label = assigned_gt_label - 1 # 1... -> 0...
            # used for dist_mask
            assigned_gt_boxes = cur_gt_boxes_3d[sampled_gt_idx] # [pts_num, 7]
            # then calc the distance between anchors and assigned_boxes
            dist = np.linalg.norm(cur_anchors_3d[:, :, :3] - assigned_gt_boxes[:, np.newaxis, :3], axis=-1) # [pts_num, cls_num]

            filtered_assigned_idx = filter_idx[sampled_gt_idx]  # [pts_num]
            filtered_assigned_idx = np.tile(np.reshape(filtered_assigned_idx, [pts_num, 1]), [1, cls_num])
            batch_assigned_idx[i] = filtered_assigned_idx

            if cls_num == 1: # anchor_free
                label_mask = np.ones([pts_num, cls_num], dtype=np.float32)
            else: # multiple anchors
                label_mask = np.tile(np.reshape(np.arange(cls_num), [1, cls_num]), [pts_num, 1])
                label_mask = np.equal(label_mask, assigned_gt_label[:, np.newaxis]).astype(np.float32)

            pmask = np.max(points_mask, axis=1) > 0
            dist_mask = np.less_equal(dist, effective_sample_range) # pts_num, cls_num
            pmask = np.logical_and(pmask[:, np.newaxis], dist_mask).astype(np.float32)
            pmask = pmask * label_mask
            pmask = pmask * cur_valid_mask

            nmask = np.max(points_mask, axis=1) == 0
            nmask = np.tile(np.reshape(nmask, [pts_num, 1]), [1, cls_num])
            nmask = nmask * label_mask
            nmask = nmask * cur_valid_mask

            # then randomly sample
            if minibatch_size != -1:
                pts_pmask = np.any(pmask, axis=1) # pts_num
                pts_nmask = np.any(nmask, axis=1) # [pts_num]

                positive_inds = np.where(pts_pmask)[0]
                cur_positive_num = np.minimum(len(positive_inds), positive_size)
                if cur_positive_num > 0:
                    positive_inds = np.random.choice(positive_inds, cur_positive_num, replace=False)
                pts_pmask = np.zeros_like(pts_pmask)
                pts_pmask[positive_inds] = 1

                cur_negative_num = minibatch_size - cur_positive_num
                negative_inds = np.where(pts_nmask)[0]
                cur_negative_num = np.minimum(len(negative_inds), cur_negative_num)
                if cur_negative_num > 0:
                    negative_inds = np.random.choice(negative_inds, cur_negative_num, replace=False)
                pts_nmask = np.zeros_like(pts_nmask)
                pts_nmask[negative_inds] = 1

                pmask = pmask * pts_pmask[:, np.newaxis]
                nmask = nmask * pts_nmask[:, np.newaxis]

            batch_assigned_pmask[i] = pmask
            batch_assigned_nmask[i] = nmask
        return batch_assigned_idx, batch_assigned_pmask, batch_assigned_nmask
