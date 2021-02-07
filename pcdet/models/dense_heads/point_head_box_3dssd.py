import torch
import torch.nn.functional as F
from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate


class PointHeadBox3DSSD(PointHeadTemplate):
    """
    Anchor-free head used in 3DSSD
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        # get coder
        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        # make prediction layers
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

        # add loss
        self.angle_bin_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.angle_res_loss = torch.nn.SmoothL1Loss(reduction='none')
        self.vote_loss = torch.nn.SmoothL1Loss(reduction='none')

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                seed_point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                aggregated_point_coords
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        aggregated_point_coords = input_dict['point_coords']
        seed_point_coords = input_dict['seed_point_coords'].detech()
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert aggregated_point_coords.shape.__len__() in [2], 'points.shape=%s' % str(aggregated_point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        # todo: implement ball constraint
        # todo: implement vote_mask
        seed_target_dict = self.assign_stack_targets(
            points=seed_point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )
        targets_dict = self.assign_stack_targets(
            points=aggregated_point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        # targets_dict['aggregated_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
        # targets_dict['aggregated_cls_labels'] = targets_dict['point_cls_labels']
        # targets_dict['aggregated_box_labels'] = targets_dict['point_box_labels']
        targets_dict['seed_gt_box_of_fg_points'] = seed_target_dict['gt_box_of_fg_points']
        targets_dict['seed_cls_labels'] = seed_target_dict['point_cls_labels']
        targets_dict['seed_box_labels'] = seed_target_dict['point_box_labels']

        return targets_dict

    def get_loss(self, tb_dict=None):
        pass

    def forward(self, batch_dict):

        aggregated_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(aggregated_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(aggregated_features)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_box_preds': point_box_preds,
            'vote_offset': batch_dict['vote_offset'],
            'seed_points': batch_dict['seed_point_coords'],
            'aggregated_points': batch_dict['point_coords']
        }

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['positive_mask'] = targets_dict['point_cls_labels'] > 0
            ret_dict['negative_mask'] = targets_dict['point_cls_labels'] == 0
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['seed_cls_labels'] = targets_dict['seed_cls_labels']
            ret_dict['seed_box_labels'] = targets_dict['seed_box_labels']
            ret_dict['seed_gt_box_of_fg_points'] = targets_dict['seed_gt_box_of_fg_points']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict

    def get_vote_loss(self, tb_dict=None):
        vote_mask = self.forward_ret_dict['seed_cls_labels'] > 0
        vote_offset = self.forward_ret_dict['vote_offset']
        seed_box_label = self.forward_ret_dict['seed_gt_box_of_fg_points'][:, 0:3]
        seed_points = self.forward_ret_dict['seed_points']
        vote_offset_label = seed_box_label - seed_points[vote_mask][:, 1:4].contiguous()

        vote_loss = self.vote_loss(vote_offset, vote_offset_label)
        vote_loss = vote_loss.sum() / (vote_mask.float().sum() + 1e-6)

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss': vote_loss.item()})
        return vote_loss, tb_dict

    def get_angle_loss(self, tb_dict=None):
        angle_bin_weight = self.forward_ret_dict['positive_mask'].float()
        angle_bin_weight = angle_bin_weight / (angle_bin_weight.sum() + 1e-6)

        point_box_labels = self.forward_ret_dict['aggregated_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']
        label_angle_bin_id = point_box_labels[:, 6].long().contiguous()
        label_angle_bin_res = point_box_labels[:, 7].contiguous()
        pred_angle_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size].contiguous()
        pred_angle_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:].contiguous()

        # bin loss
        angle_bin_loss = self.angle_bin_loss(pred_angle_bin_id, label_angle_bin_id)
        angle_bin_loss = torch.sum(angle_bin_loss * angle_bin_weight)
        # res loss
        # todo: examine output of head
        label_angle_bin_id_onehot = F.one_hot(label_angle_bin_id.long().contiguous(), self.box_coder.bin_size)
        pred_angle_bin_res = torch.sum(pred_angle_bin_res * label_angle_bin_id_onehot.float(), dim=-1)
        angle_res_loss = self.angle_res_loss(pred_angle_bin_res, label_angle_bin_res)
        angle_res_loss = torch.sum(angle_res_loss * angle_bin_weight)

        angle_loss = angle_res_loss + angle_bin_loss

        tb_dict.update({'angle_res_loss': angle_res_loss.item()})
        tb_dict.update({'angle_bin_loss': angle_bin_loss.item()})
        tb_dict.update({'angle_loss': angle_loss.item()})

        return angle_loss, tb_dict

    def get_corner_loss(self, tb_dict=None):

        pass

    def get_box_layer_loss(self, tb_dict=None):
        box_res_weight = self.forward_ret_dict['positive_mask'].float()
        box_res_weight = box_res_weight / (box_res_weight.sum() + 1e-6)

        point_box_labels = self.forward_ret_dict['aggregated_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        box_res_loss = self.reg_loss_func(
            pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=box_res_weight[None, ...]
        )
        box_res_loss = torch.sum(box_res_loss)

        tb_dict.update({'box_res_loss': box_res_loss.item()})

        return box_res_loss, tb_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positive_mask = self.forward_ret_dict['positive_mask']
        negative_mask = self.forward_ret_dict['negative_mask']
        cls_weight = (positive_mask.float() + negative_mask.float())
        cls_weight = cls_weight / (positive_mask.sum(dim=0) + 1e-6)

        point_cls_labels_onehot = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        point_cls_labels_onehot.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1), 1)
        point_cls_labels_onehot = point_cls_labels_onehot[..., 1:].contiguous()

        # calculate centerness
        centerness_mask = self._generate_centerness_label()
        point_cls_labels_onehot = point_cls_labels_onehot * centerness_mask.unsqueeze(-1).repeat(1, point_cls_labels_onehot.shape[1])
        cls_loss_src = loss_utils.SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits(point_cls_preds, one_hot_targets)
        cls_loss_src = cls_loss_src * cls_weight.unsqueeze(-1)

    def _generate_centerness_label(self):
        pass



