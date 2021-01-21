import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.core.config import cfg
from lib.builder.anchor_builder import Anchors
from lib.builder.encoder_builder import EncoderDecoder
from lib.builder.layer_builder import LayerBuilder
from lib.modeling.head_builder import HeadBuilder
from lib.utils.model_util import merge_head_prediction
from lib.builder.target_assigner import TargetAssigner
from lib.builder.loss_builder import LossBuilder
from lib.builder.postprocessor import PostProcessor
import os


import lib.dataset.maps_dict as maps_dict

from lib.utils.box_3d_utils import transfer_box3d_to_corners_torch


# TODO: put bn_decay into init


class SingleStageDetector(nn.Module):
    def __init__(self, batch_size, is_training):
        super().__init__()
        self.batch_size = batch_size
        self.is_training = is_training

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.cls2idx = dict([(cls, i + 1) for i, cls in enumerate(self.cls_list)])
        self.idx2cls = dict([(i + 1, cls) for i, cls in enumerate(self.cls_list)])

        # anchor_builder
        self.anchor_builder = Anchors(0, self.cls_list)

        # encoder_decoder
        self.encoder_decoder = EncoderDecoder(0)

        self.corner_loss = cfg.MODEL.FIRST_STAGE.CORNER_LOSS

        # layer builder
        self.vote_loss = False
        layer_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.ARCHITECTURE
        layers = []
        for i in range(len(layer_cfg)):
            layers.append(LayerBuilder(i, self.is_training, layer_cfg))
            if layers[-1].layer_type == 'Vote_Layer':
                self.vote_loss = True
        self.layers = nn.Sequential(*layers)

        # head builder
        self.iou_loss = False
        heads = []
        head_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.HEAD
        for i in range(len(head_cfg)):
            heads.append(HeadBuilder(self.batch_size,
                                          self.anchor_builder.anchors_num, 0, head_cfg[i], is_training))
            if heads[-1].layer_type == 'IoU':
                self.iou_loss = True
        self.heads = nn.Sequential(*heads)

        # target assigner
        self.target_assigner = TargetAssigner(0)  # first stage

        self.attr_velo_loss = cfg.MODEL.FIRST_STAGE.PREDICT_ATTRIBUTE_AND_VELOCITY

    def __init_dict(self, batch_data_label=None):
        end_points = dict()

        # sampled xyz/feature
        end_points[maps_dict.KEY_OUTPUT_XYZ] = []
        end_points[maps_dict.KEY_OUTPUT_FEATURE] = []
        # generated anchors
        end_points[maps_dict.KEY_ANCHORS_3D] = []  # generated anchors
        # vote output
        end_points[maps_dict.PRED_VOTE_OFFSET] = []
        end_points[maps_dict.PRED_VOTE_BASE] = []
        # det output
        end_points[maps_dict.PRED_CLS] = []
        end_points[maps_dict.PRED_OFFSET] = []
        end_points[maps_dict.PRED_ANGLE_CLS] = []
        end_points[maps_dict.PRED_ANGLE_RES] = []
        end_points[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS] = []
        end_points[maps_dict.PRED_ATTRIBUTE] = []
        end_points[maps_dict.PRED_VELOCITY] = []
        # iou output
        end_points[maps_dict.PRED_IOU_3D_VALUE] = []
        # final result
        end_points[maps_dict.PRED_3D_BBOX] = []
        end_points[maps_dict.PRED_3D_SCORE] = []
        end_points[maps_dict.PRED_3D_CLS_CATEGORY] = []
        end_points[maps_dict.PRED_3D_ATTRIBUTE] = []
        end_points[maps_dict.PRED_3D_VELOCITY] = []

        # self.prediction_keys = self.output.keys()

        # self.labels = dict()
        end_points[maps_dict.GT_CLS] = []
        end_points[maps_dict.GT_OFFSET] = []
        end_points[maps_dict.GT_ANGLE_CLS] = []
        end_points[maps_dict.GT_ANGLE_RES] = []
        end_points[maps_dict.GT_ATTRIBUTE] = []
        end_points[maps_dict.GT_VELOCITY] = []
        end_points[maps_dict.GT_BOXES_ANCHORS_3D] = []
        end_points[maps_dict.GT_IOU_3D_VALUE] = []

        end_points[maps_dict.GT_PMASK] = []
        end_points[maps_dict.GT_NMASK] = []
        end_points[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS] = []

        end_points['offset_loss'] = []

        # self.batch_data_label = dict()
        if batch_data_label is not None:
            end_points[maps_dict.PL_POINTS_INPUT] = batch_data_label[maps_dict.PL_POINTS_INPUT]
            end_points[maps_dict.PL_LABEL_BOXES_3D] = batch_data_label[maps_dict.PL_LABEL_BOXES_3D]
            end_points[maps_dict.PL_LABEL_CLASSES] = batch_data_label[maps_dict.PL_LABEL_CLASSES]
            end_points[maps_dict.PL_ANGLE_CLS] = batch_data_label[maps_dict.PL_ANGLE_CLS]
            end_points[maps_dict.PL_ANGLE_RESIDUAL] = batch_data_label[maps_dict.PL_ANGLE_RESIDUAL]
        return end_points


    def forward(self, batch, bn_decay=None):

        end_points = self.__init_dict(batch_data_label=batch)

        points_input_det = batch['point_cloud_pl']

        # forward the point cloud
        end_points = self.network_forward(points_input_det, end_points)

        # generate anchors
        base_xyz = end_points[maps_dict.KEY_OUTPUT_XYZ][-1]
        anchors = self.anchor_builder.generate(base_xyz)  # [bs, pts_num, 1/cls_num, 7]
        end_points[maps_dict.KEY_ANCHORS_3D].append(anchors)

        if self.is_training:  # training mode
            end_points = self.train_forward(-1, anchors, end_points)
            return end_points
        else:  # testing mode
            end_points = self.test_forward(-1, anchors, end_points)
            return end_points

    def network_forward(self, point_cloud, end_points):
        l0_xyz = point_cloud[:, :, 0:3]
        l0_points = point_cloud[:, :, 3:]
        xyz_list, feature_list, fps_idx_list = [l0_xyz], [l0_points], [None]
        for layer in self.layers:
            xyz_list, feature_list, fps_idx_list, end_points = layer(xyz_list, feature_list, fps_idx_list, end_points)

        cur_head_start_idx = -1
        for head in self.heads:
            xyz_output, feature_output, end_points = head(xyz_list, feature_list, end_points)
        # merge_head_prediction(cur_head_start_idx, self.output, self.prediction_keys)
        return end_points

    def train_forward(self, index, anchors, end_points):
        base_xyz = end_points[maps_dict.KEY_OUTPUT_XYZ][index]
        pred_offset = end_points[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = end_points[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = end_points[maps_dict.PRED_ANGLE_RES][index]

        gt_boxes_3d = end_points[maps_dict.PL_LABEL_BOXES_3D]
        gt_classes = end_points[maps_dict.PL_LABEL_CLASSES]
        gt_angle_cls = end_points[maps_dict.PL_ANGLE_CLS]
        gt_angle_res = end_points[maps_dict.PL_ANGLE_RESIDUAL]

        returned_list = self.target_assigner.assign(base_xyz, anchors, gt_boxes_3d, gt_classes, gt_angle_cls, gt_angle_res)

        assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute = returned_list

        # encode offset
        assigned_gt_offset, assigned_gt_angle_cls, assigned_gt_angle_res = self.encoder_decoder.encode(base_xyz,
                                                                                                       assigned_gt_boxes_3d,
                                                                                                       anchors)

        # corner_loss
        corner_loss_angle_cls = nn.functional.one_hot(assigned_gt_angle_cls, num_classes=cfg.MODEL.ANGLE_CLS_NUM)  # bs, pts_num, cls_num, -1
        pred_anchors_3d = self.encoder_decoder.decode(base_xyz, pred_offset, corner_loss_angle_cls, pred_angle_res,
                                                      self.is_training, anchors)  # [bs, points_num, cls_num, 7]
        pred_corners = transfer_box3d_to_corners_torch(pred_anchors_3d)  # [bs, points_num, cls_num, 8, 3]
        gt_corners = transfer_box3d_to_corners_torch(assigned_gt_boxes_3d)  # [bs, points_num, cls_num,8,3]

        end_points[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS].append(pred_corners)
        end_points[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS].append(gt_corners)

        end_points[maps_dict.GT_CLS].append(assigned_gt_labels)
        end_points[maps_dict.GT_BOXES_ANCHORS_3D].append(assigned_gt_boxes_3d)
        end_points[maps_dict.GT_OFFSET].append(assigned_gt_offset)
        end_points[maps_dict.GT_ANGLE_CLS].append(assigned_gt_angle_cls)
        end_points[maps_dict.GT_ANGLE_RES].append(assigned_gt_angle_res)
        end_points[maps_dict.GT_ATTRIBUTE].append(assigned_gt_attribute)
        end_points[maps_dict.GT_VELOCITY].append(assigned_gt_velocity)
        end_points[maps_dict.GT_PMASK].append(assigned_pmask)
        end_points[maps_dict.GT_NMASK].append(assigned_nmask)

        return end_points

        # total_loss, loss_dict, end_points = self.loss_builder(index, end_points, self.corner_loss, self.vote_loss, self.attr_velo_loss, self.iou_loss, disp_dict)
        # disp_dict.update(loss_dict)
        #
        # return total_loss, end_points, disp_dict

    def test_forward(self, index, anchors, end_points):
        base_xyz = end_points[maps_dict.KEY_OUTPUT_XYZ][index]

        pred_cls = end_points[maps_dict.PRED_CLS][index]  # [bs, points_num, cls_num + 1/0]
        pred_offset = end_points[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = end_points[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = end_points[maps_dict.PRED_ANGLE_RES][index]

        # decode predictions
        pred_anchors_3d = self.encoder_decoder.decode(base_xyz, pred_offset, pred_angle_cls, pred_angle_res,
                                                      self.is_training, anchors)  # [bs, points_num, cls_num, 7]

        # decode classification
        if cfg.MODEL.FIRST_STAGE.CLS_ACTIVATION == 'Softmax':
            assert cfg.MODEL.FIRST_STAGE.CLS_ACTIVATION == 'Sigmoid'
            # softmax
            # pred_score = tf.nn.softmax(pred_cls)
            # pred_score = tf.slice(pred_score, [0, 0, 1], [-1, -1, -1])
        else: # sigmoid
            pred_score = torch.sigmoid(pred_cls)

        end_points['post_pred_score'] = pred_score
        end_points['pred_anchors_3d'] = pred_anchors_3d

        return end_points


    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))


def compute_loss(end_points):
    loss_builder = LossBuilder(0)
    index = -1
    total_loss, loss_dict = loss_builder.compute_loss(index, end_points, corner_loss_flag=True, vote_loss_flag=True)

    return total_loss, loss_dict


def post_process(end_points):
    # postprocessor
    postprocessor = PostProcessor(0, len(cfg.DATASET.KITTI.CLS_LIST))

    pred_score = end_points['post_pred_score']
    pred_anchors_3d = end_points['pred_anchors_3d']

    end_points = postprocessor.forward(pred_anchors_3d, pred_score, end_points)
    return end_points

