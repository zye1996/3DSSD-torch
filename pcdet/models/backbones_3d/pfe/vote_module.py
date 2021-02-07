import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_batch import pointnet2_modules
from ....ops.pointnet2.pointnet2_batch import pointnet2_modules as pointnet2_batch_modules

from ....utils import common_utils


class VoteModule(nn.Module):
    
    def __init__(self, model_cfg, voxel_size=None, point_cloud_range=None, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):

        super().__init__()
        self.model_cfg = model_cfg
        self.num_points = model_cfg.NUM_POINTS
        self.in_channels = model_cfg.NUM_INPUT_FEATURES
        self.vote_xyz_range = model_cfg.VOTE_RANGE
        self.with_res_feat = model_cfg.WITH_RES_FEATURE

        self.aggre_mlps = model_cfg.AGGREGATION_MLPS
        self.aggre_radius = model_cfg.AGGREGATION_RADIUS
        self.aggre_samples = model_cfg.AGGREGATION_NSAMPLES
        # self.skip_channels = model_cfg.SKIP_CHANNELS

        # vote mlps
        mlp = model_cfg.MLPS
        mlp = [self.in_channels] + mlp

        vote_conv_list = list()
        for k in range(len(mlp) - 1):
            vote_conv_list.extend([
                nn.Conv1d(mlp[k], mlp[k + 1], kernel_size=1, bias=True),
                nn.BatchNorm1d(mlp[k + 1]),
                nn.ReLU()
            ])

        if self.with_res_feat:
            out_channel = 3 + self.in_channels
        else:
            out_channel = 3

        vote_conv_list.extend([
            nn.Conv1d(mlp[-1], out_channels=out_channel, kernel_size=1),
        ])

        self.vote_mlp = nn.Sequential(*vote_conv_list)

        # aggregation
        self.vote_aggregation = pointnet2_batch_modules.PointnetSAModuleMSG_FPS(npoint=self.num_points,
                                                                                radii=self.aggre_radius,
                                                                                nsamples=self.aggre_samples,
                                                                                mlps=self.aggre_mlps,
                                                                                use_xyz=True)

        sa_channel_out = 0
        for aggre_mlp in self.aggre_mlps:
            sa_channel_out += aggre_mlp[-1]

        self.conv_out = nn.Sequential(
                        nn.Conv1d(sa_channel_out, self.model_cfg.AGGREGATION_OUT, kernel_size=1, bias=False),
                        nn.BatchNorm1d(self.model_cfg.AGGREGATION_OUT),
                        nn.ReLU())

        # TODO: optional FP module for PointRCNN compatibility
        '''
        self.FP_modules = nn.ModuleList()
        channel_out = self.model_cfg.AGGREGATION_OUT

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + self.skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )
        '''

    def extract_input(self, batch_dict):
        batch_size = batch_dict['batch_size']

        xyz = batch_dict['point_coords'].view(batch_size, -1, 4)[..., 1:].contiguous()
        features = batch_dict['point_features'].view(batch_size, -1, batch_dict['point_features'].shape[-1]).contiguous()

        return xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """

        batch_size = batch_dict['batch_size']
        xyz, features = self.extract_input(batch_dict)
        features = features.permute(0, 2, 1).contiguous()

        if isinstance(self.num_points, list):
            self.num_points = self.num_points[0]

        seed_points = xyz[:, :self.num_points, :].contiguous()  # (B, N, 3)
        seed_features = features[:, :, :self.num_points].contiguous()  # (B, C, N)

        # generate vote points
        votes = self.vote_mlp(seed_features)  # (B, 3+C, N)
        votes = votes.transpose(2, 1)

        seed_offset = votes[:, :, :3]
        limited_offset_list = []
        for axis in range(len(self.vote_xyz_range)):
            limited_offset_list.append(
                seed_offset[..., axis].clamp(min=-self.vote_xyz_range[axis],
                                             max=self.vote_xyz_range[axis])
            )
        limited_offset = torch.stack(limited_offset_list, dim=-1)  # (B, N, 3)
        vote_points = (seed_points + limited_offset).contiguous()

        # generate shifted features
        if self.with_res_feat:
            res_features = votes[:, :, 3:]
            vote_features = res_features + seed_features.transpose(2, 1).contiguous()
        else:
            vote_features = seed_features.transpose(2, 1).contiguous()

        # aggregation
        aggregated_points, aggregated_features = self.vote_aggregation(xyz=xyz,
                                                                       features=features,
                                                                       new_xyz=vote_points)
        aggregated_features = self.conv_out(aggregated_features)

        # FP forward


        # pack output
        ctr_batch_idx = torch.arange(batch_size, device=seed_offset.device).view(-1, 1).repeat(1, seed_offset.shape[1]).view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), seed_offset.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), aggregated_points.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), seed_points.contiguous().view(-1, 3)), dim=1)
        center_features = aggregated_features.permute(0, 2, 1).contiguous().view(-1, aggregated_features.shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx

        return batch_dict

