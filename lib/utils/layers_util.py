import tensorflow as tf
import numpy as np
# import utils.tf_util as tf_util
# import utils.model_util as model_util

import torch.nn as nn
import torch
import torch.nn.functional as F
import lib.pointnet2.pointnet2_utils as pointnet2_utils
import lib.pointnet2.pytorch_utils as pt_utils
from lib.utils.model_util import calc_square_dist, nn_distance


# from utils.tf_ops.grouping.tf_grouping import *
# from utils.tf_ops.sampling.tf_sampling import *
# from utils.tf_ops.interpolation.tf_interpolate import *
from lib.core.config import cfg


def vote_layer_funciton(xyz, points, mlp_list, is_training, bn_decay, bn, scope):
    """
    Voting layer
    """
    with tf.variable_scope(scope) as sc:
        for i, channel in enumerate(mlp_list):
            points = tf_util.conv1d(points, channel, 1, padding='VALID', stride=1, bn=bn, scope='vote_layer_%d'%i, bn_decay=bn_decay, is_training=is_training)
        ctr_offsets = tf_util.conv1d(points, 3, 1, padding='VALID', stride=1, bn=False, activation_fn=None, scope='vote_offsets')

        min_offset = tf.reshape(cfg.MODEL.MAX_TRANSLATE_RANGE, [1, 1, 3])
        limited_ctr_offsets = tf.minimum(tf.maximum(ctr_offsets, min_offset), -min_offset)
        xyz = xyz + limited_ctr_offsets 
    return xyz, points, ctr_offsets


class Vote_layer(nn.Module):
    def __init__(self, mlp_list, bn, is_training, pre_channel):
        super().__init__()
        self.mlp_list = mlp_list
        self.bn = bn
        self.is_training = is_training

        mlp_modules = []
        for i in range(len(self.mlp_list)):
            mlp_modules.append(pt_utils.Conv1d(pre_channel, self.mlp_list[i], bn=self.bn))
            pre_channel = self.mlp_list[i]
        self.mlp_modules = nn.Sequential(*mlp_modules)

        self.ctr_reg = pt_utils.Conv1d(pre_channel, 3, activation=None, bn=False)
        self.min_offset = torch.tensor(cfg.MODEL.MAX_TRANSLATE_RANGE).float().view(1, 1, 3)

    def forward(self, xyz, points):

        points_transpose = points.transpose(1, 2)

        points_transpose = self.mlp_modules(points_transpose)
        ctr_offsets = self.ctr_reg(points_transpose)

        ctr_offsets = ctr_offsets.transpose(1, 2)
        points = points_transpose.transpose(1, 2)

        min_offset = torch.tensor(cfg.MODEL.MAX_TRANSLATE_RANGE).float().view(1, 1, 3).repeat((points.shape[0], points.shape[1], 1)).to(points.device)

        limited_ctr_offsets = torch.where(ctr_offsets < min_offset, min_offset, ctr_offsets)
        min_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(limited_ctr_offsets > min_offset, min_offset, limited_ctr_offsets)
        xyz = xyz + limited_ctr_offsets
        return xyz, points, ctr_offsets


def pointnet_sa_module(xyz, points, mlp,
                       is_training, bn_decay, bn, 
                       scope):
    ''' PointNet Set Abstraction (SA) Module (Last Layer)
        Sample all points within the point cloud and extract a global feature
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            mlp_list: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        grouped_points = tf.concat([xyz, points], axis=-1) # [bs, npoint, 3+c] 

        for j, num_out_channel in enumerate(mlp):
            grouped_points = tf_util.conv1d(grouped_points, 
                                            num_out_channel, 
                                            1,
                                            padding='VALID', 
                                            bn=bn, 
                                            is_training=is_training,
                                            scope='conv%d' % j, 
                                            bn_decay=bn_decay)
        # bs, num_out_channel
        new_points = tf.reduce_max(grouped_points, axis=1)
    return new_points
    


class Pointnet_sa_module_msg(nn.Module):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int -- points sampled in farthest point sampling
            radius_list: list of float32 -- search radius in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp_list: list of list of int32 -- output size for MLP on each point
            fps_method: 'F-FPS', 'D-FPS', 'FS'
            fps_start_idx:
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''

    def __init__(self, radius_list, nsample_list,
                 mlp_list, is_training, bn_decay, bn,
                 fps_sample_range_list, fps_method_list, npoint_list, use_attention, scope,
                 dilated_group, aggregation_channel=None, pre_channel=0,
                 debugging=False,
                 epsilon=1e-5):
        super().__init__()

        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_list = mlp_list
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.fps_sample_range_list = fps_sample_range_list
        self.fps_method_list = fps_method_list
        self.npoint_list = npoint_list
        self.use_attention = use_attention
        self.scope = scope
        self.dilated_group = dilated_group
        self.aggregation_channel = aggregation_channel
        self.pre_channel = pre_channel

        mlp_modules = []
        for i in range(len(self.radius_list)):
            mlp_spec = [self.pre_channel + 3] + self.mlp_list[i]
            mlp_modules.append(pt_utils.SharedMLP(mlp_spec, bn=self.bn))
        self.mlp_modules = nn.Sequential(*mlp_modules)

        if cfg.MODEL.NETWORK.AGGREGATION_SA_FEATURE and (len(self.mlp_list) != 0):
            input_channel = 0
            for mlp_tmp in self.mlp_list:
                input_channel += mlp_tmp[-1]
            self.aggregation_layer = pt_utils.Conv1d(input_channel, aggregation_channel, bn=self.bn)

    def forward(self, xyz, points, former_fps_idx, vote_ctr):
        bs = xyz.shape[0]
        num_points = xyz.shape[1]

        cur_fps_idx_list = []
        last_fps_end_index = 0
        for fps_sample_range, fps_method, npoint in zip(self.fps_sample_range_list, self.fps_method_list, self.npoint_list):
            if fps_sample_range < 0:
                fps_sample_range_tmp = fps_sample_range + num_points + 1
            else:
                fps_sample_range_tmp = fps_sample_range
            tmp_xyz = xyz[:, last_fps_end_index:fps_sample_range_tmp, :].contiguous()
            tmp_points = points[:, last_fps_end_index:fps_sample_range_tmp, :].contiguous()
            if npoint == 0:
                last_fps_end_index += fps_sample_range
                continue
            if vote_ctr is not None:
                npoint = vote_ctr.shape[1]
                fps_idx = torch.arange(npoint).int().view(1, npoint).repeat((bs, 1)).to(tmp_xyz.device)
            elif fps_method == 'FS':
                features_for_fps = torch.cat([tmp_xyz, tmp_points], dim=-1)
                # dist1 = nn_distance(tmp_xyz, tmp_xyz)
                # dist2 = calc_square_dist(tmp_xyz, tmp_xyz, norm=False)
                features_for_fps_distance = calc_square_dist(features_for_fps, features_for_fps, norm=False)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                fps_idx_2 = pointnet2_utils.furthest_point_sample(tmp_xyz, npoint)
                fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)  # [bs, npoint * 2]
            elif npoint == tmp_xyz.shape[1]:
                fps_idx = torch.arange(npoint).int().view(1, npoint).repeat((bs, 1)).to(tmp_xyz.device)
            elif fps_method == 'F-FPS':
                features_for_fps = torch.cat([tmp_xyz, tmp_points], dim=-1)
                features_for_fps_distance = calc_square_dist(features_for_fps, features_for_fps, norm=False)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
            else: # D-FPS
                fps_idx = pointnet2_utils.furthest_point_sample(tmp_xyz, npoint)

            fps_idx = fps_idx + last_fps_end_index
            cur_fps_idx_list.append(fps_idx)
            last_fps_end_index += fps_sample_range
        fps_idx = torch.cat(cur_fps_idx_list, dim=-1)

        if former_fps_idx is not None:
            fps_idx = torch.cat([fps_idx, former_fps_idx], dim=-1)

        if vote_ctr is not None:
            vote_ctr_transpose = vote_ctr.transpose(1, 2).contiguous()
            new_xyz = pointnet2_utils.gather_operation(vote_ctr_transpose, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        # # if deformed_xyz is not None, then no attention model
        # if use_attention:
        #     # first gather the points out
        #     new_points = gather_point(points, fps_idx) # [bs, npoint, c]
        #
        #     # choose farthest feature to center points
        #     # [bs, npoint, ndataset]
        #     relation = model_util.calc_square_dist(new_points, points)
        #     # choose these points with largest distance to center_points
        #     _, relation_idx = tf.nn.top_k(relation, k=relation.shape.as_list()[-1])

        new_points_list = []
        points = points.transpose(1, 2).contiguous()
        xyz = xyz.contiguous()
        for i in range(len(self.radius_list)):
            nsample = self.nsample_list[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.0
                else:
                    min_radius = self.radius_list[i-1]
                max_radius = self.radius_list[i]
                idx = pointnet2_utils.ball_query_dilated(max_radius, min_radius, nsample, xyz, new_xyz)
            else:
                radius = self.radius_list[i]
                idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)

            xyz_trans = xyz.transpose(1, 2).contiguous()
            grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
            grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
            if points is not None:
                grouped_points = pointnet2_utils.grouping_operation(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                grouped_points = grouped_xyz

            new_points = self.mlp_modules[i](grouped_points)
            new_points = F.max_pool2d(new_points, kernel_size=[1, new_points.size(3)])
            new_points_list.append(new_points.squeeze(-1))

        if len(new_points_list) > 0:
            new_points_concat = torch.cat(new_points_list, dim=1)
            if cfg.MODEL.NETWORK.AGGREGATION_SA_FEATURE:
                new_points_concat = self.aggregation_layer(new_points_concat)
        else:
            new_points_concat = pointnet2_utils.gather_operation(points, fps_idx)
        new_points_concat = new_points_concat.transpose(1, 2).contiguous()

        return new_xyz, new_points_concat, fps_idx


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            the unknown features 13
            xyz1: (batch_size, ndataset1, 3) TF tensor
            the known features 14
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1


# if __name__ == '__main__':
#
#     xyz = torch.rand((2, 4096, 3)).cuda()
#     features = torch.rand((2, 32, 4096)).cuda()
#     sa_msg = a