import os
import numpy as np

import torch.utils.data as torch_data
from lib.core.config import cfg
import lib.utils.kitti_object as kitti_object
import lib.dataset.maps_dict as maps_dict
from lib.utils.anchor_encoder import encode_angle2class_np
from lib.builder.data_augmentor import DataAugmentor
from itertools import chain
import torch
from lib.utils.voxelnet_aug import check_inside_points
import cv2
from lib.viz.viz_utils import point_viz
from lib.utils.box_3d_utils import get_box3d_corners_helper_np
from lib.utils.anchors_util import project_to_image_space_corners
from lib.utils.points_filter import get_point_filter, get_point_filter_in_image
from lib.utils.box_3d_utils import object_label_to_box_3d
import tqdm
import copy


class KittiDataset(torch_data.Dataset):

    def __init__(self, mode, split='training', img_list='trainval', is_training=True):
        """
        mode: 'loading', 'preprocessing'
        """
        super().__init__()
        self.mode = mode
        self.dataset_dir = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.BASE_DIR_PATH)
        self.label_dir = os.path.join(cfg.DATASET.KITTI.BASE_DIR_PATH, split, 'label_2')
        self.kitti_object = kitti_object.kitti_object(self.dataset_dir, split)
        self.is_training = is_training
        self.img_list = img_list

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.idx2cls_dict = dict([(idx+1, cls) for idx, cls in enumerate(self.cls_list)])
        self.cls2idx_dict = dict([(cls, idx+1) for idx, cls in enumerate(self.cls_list)])

        # formulate save data_dir
        base_dir = ''
        if not cfg.TEST.WITH_GT:
            base_dir += 'no_gt/'
        self.sv_npy_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.SAVE_NUMPY_PATH, base_dir + self.img_list,
                                        '{}'.format(self.cls_list))

        self.train_list = os.path.join(self.sv_npy_path, 'train_list.txt')

        self.test_mode = cfg.TEST.TEST_MODE
        # if self.test_mode == 'mAP':
        #     self.evaluation = self.evaluate_map
        #     self.logger_and_select_best = self.logger_and_select_best_map
        # elif self.test_mode == 'Recall':
        #     self.evaluation = self.evaluate_recall
        #     self.logger_and_select_best = self.logger_and_select_best_recall
        # else:
        #     raise Exception('No other evaluation mode.')

        if mode == 'loading':
            # data loader
            with open(self.train_list, 'r') as f:
                self.train_npy_list = [line.strip('\n') for line in f.readlines()]
            # self.train_npy_list = self.train_npy_list[0:100]
            self.train_npy_list = np.array(self.train_npy_list)
            self.sample_num = len(self.train_npy_list)
            if self.is_training:
                self.data_augmentor = DataAugmentor('KITTI')

        elif mode == 'preprocessing':
            # preprocess raw data
            if img_list == 'train':
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.TRAIN_LIST)
            elif img_list == 'val':
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.VAL_LIST)
            elif img_list == 'trainval':
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.TRAINVAL_LIST)
            else:
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.TEST_LIST)
            with open(list_path, 'r') as f:
                self.idx_list = [line.strip('\n') for line in f.readlines()]
            self.sample_num = len(self.idx_list)

            self.extents = cfg.DATASET.POINT_CLOUD_RANGE
            self.extents = np.reshape(self.extents, [3, 2])
            if not os.path.exists(self.sv_npy_path): os.makedirs(self.sv_npy_path)

            # the save path for MixupDB
            if self.img_list in ['train', 'val',
                                 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
                self.mixup_db_cls_path = dict()
                self.mixup_db_trainlist_path = dict()
                self.mixup_db_class = cfg.TRAIN.AUGMENTATIONS.MIXUP.CLASS
                for cls in self.mixup_db_class:
                    mixup_db_cls_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.SAVE_NUMPY_PATH,
                                                     cfg.TRAIN.AUGMENTATIONS.MIXUP.SAVE_NUMPY_PATH,
                                                     cfg.TRAIN.AUGMENTATIONS.MIXUP.PC_LIST, '{}'.format(cls))
                    mixup_db_trainlist_path = os.path.join(mixup_db_cls_path, 'train_list.txt')
                    if not os.path.exists(mixup_db_cls_path): os.makedirs(mixup_db_cls_path)
                    self.mixup_db_cls_path[cls] = mixup_db_cls_path
                    self.mixup_db_trainlist_path[cls] = mixup_db_trainlist_path

    def __len__(self):
        return self.sample_num

    def __getitem__(self, sample_idx):
        samples = self.load_samples(sample_idx)
        results = {
            'cur_biggest_label': samples[0],
            'samples': samples[1:]
        }

        return results

    def load_samples(self, sample_idx):
        """ load data per thread """
        biggest_label_num = 0
        cur_npy = self.train_npy_list[sample_idx]

        cur_npy_path = os.path.join(self.sv_npy_path, cur_npy)
        sample_dict = np.load(cur_npy_path, allow_pickle=True).tolist()

        sem_labels = sample_dict[maps_dict.KEY_LABEL_SEMSEG]
        sem_dists = sample_dict[maps_dict.KEY_LABEL_DIST]
        points = sample_dict[maps_dict.KEY_POINT_CLOUD]
        calib = sample_dict[maps_dict.KEY_STEREO_CALIB]

        if self.is_training or cfg.TEST.WITH_GT:
            label_boxes_3d = sample_dict[maps_dict.KEY_LABEL_BOXES_3D]
            label_classes = sample_dict[maps_dict.KEY_LABEL_CLASSES]
            cur_label_num = sample_dict[maps_dict.KEY_LABEL_NUM]
            ry_cls_label, residual_angle = encode_angle2class_np(label_boxes_3d[:, -1],
                                                                 num_class=cfg.MODEL.ANGLE_CLS_NUM)
        else:
            label_boxes_3d = np.zeros([1, 7], np.float32)
            label_classes = np.zeros([1], np.int32)
            cur_label_num = 1
            ry_cls_label = np.zeros([1], np.int32)
            residual_angle = np.zeros([1], np.float32)

        if self.is_training:  # then add data augmentation here
            # get plane first
            sample_name = sample_dict[maps_dict.KEY_SAMPLE_NAME]
            plane = self.kitti_object.get_planes(sample_name)
            points, sem_labels, sem_dists, label_boxes_3d, label_classes = self.data_augmentor.kitti_forward(points,
                                                                                                             sem_labels,
                                                                                                             sem_dists,
                                                                                                             label_boxes_3d,
                                                                                                             label_classes,
                                                                                                             plane)
            cur_label_num = len(label_boxes_3d)
            ry_cls_label, residual_angle = encode_angle2class_np(label_boxes_3d[:, -1],
                                                                 num_class=cfg.MODEL.ANGLE_CLS_NUM)

        # randomly choose points
        pts_num = points.shape[0]
        pts_idx = np.arange(pts_num)
        if pts_num >= cfg.MODEL.POINTS_NUM_FOR_TRAINING:
            sampled_idx = np.random.choice(pts_idx, cfg.MODEL.POINTS_NUM_FOR_TRAINING, replace=False)
        else:
            # pts_num < model_util.points_num_for_training
            # first random choice pts_num, replace=False
            sampled_idx_1 = np.random.choice(pts_idx, pts_num, replace=False)
            sampled_idx_2 = np.random.choice(pts_idx, cfg.MODEL.POINTS_NUM_FOR_TRAINING - pts_num, replace=True)
            sampled_idx = np.concatenate([sampled_idx_1, sampled_idx_2], axis=0)

        sem_labels = sem_labels[sampled_idx]
        sem_dists = sem_dists[sampled_idx]
        points = points[sampled_idx, :]

        biggest_label_num = max(biggest_label_num, cur_label_num)

        # points_mask = check_inside_points(points, label_boxes_3d)
        # points_inside = points[np.any(points_mask, axis=-1)]

        # points_mask_torch = torch.from_numpy(points_mask).float()
        # a = np.argmax(points_mask, axis=-1)
        # b = torch.argmax(points_mask_torch, dim=-1)

        # point_viz(points_inside, name='{}_point_inside'.format(sample_idx), dump_dir='viz_dump')

        return biggest_label_num, points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle, label_classes, calib.P, \
               sample_dict[maps_dict.KEY_SAMPLE_NAME]

    def load_batch(self, batch):
        use_list = False
        use_concat = [0, 0, 0, 2, 2, 2, 2, 0, 0]

        data_holder = []
        cur_biggest_label = -1
        for tmp in batch:
            if tmp['cur_biggest_label'] > cur_biggest_label:
                cur_biggest_label = tmp['cur_biggest_label']
            data_holder.append(tmp['samples'])

        size = len(data_holder[0])
        if not (isinstance(use_list, list) or isinstance(use_list, tuple)):
            use_list = [use_list for i in range(size)]
        if not (isinstance(use_concat, list) or isinstance(use_concat, tuple)):
            use_concat = [use_concat for i in range(size)]

        result = []
        for k in range(size):
            if use_concat[k]:
                dt = data_holder[0][k]
                if isinstance(dt, list):
                    result.append(
                        list(chain(*[x[k] for x in data_holder])))
                else:
                    try:
                        if use_concat[k] is True or use_concat[k] == 1:
                            result.append(
                                np.concatenate([x[k] for x in data_holder], axis=0))
                        elif use_concat[k] == 3:  # concatenate groundtruth
                            if len(data_holder[0][k].shape) == 3:  # label_boxes_3d, label_anchors...
                                result.append(
                                    np.concatenate([np.pad(x[k],
                                                           ((0, 0), (0, cur_biggest_label - x[k].shape[1]), (0, 0)),
                                                           mode='constant', constant_values=0) for i, x in
                                                    enumerate(data_holder)], axis=0))
                            elif len(data_holder[0][k].shape) == 2:  # label_classes ...
                                result.append(
                                    np.concatenate([np.pad(x[k], ((0, 0), (0, cur_biggest_label - x[k].shape[1])),
                                                           mode='constant', constant_values=0) for i, x in
                                                    enumerate(data_holder)], axis=0))
                            else:
                                raise ValueError('Unsupported type of attribute use_concat : {}'.format(use_concat[k]))
                        else:
                            if len(data_holder[0][k].shape) == 2:  # label_boxes_3d, label_anchors
                                if use_concat[k] == 2:
                                    result.append(
                                        np.stack([np.pad(x[k], ((0, cur_biggest_label - len(x[k])), (0, 0)),
                                                         mode='constant', constant_values=0) for i, x in
                                                  enumerate(data_holder)], axis=0))
                                else:
                                    raise ValueError(
                                        'Unsupported type of attribute use_concat : {}'.format(use_concat[k]))
                            elif len(data_holder[0][k].shape) == 1:  # label_class...
                                if use_concat[k] == 2:
                                    result.append(
                                        np.stack([np.pad(x[k], (0, cur_biggest_label - len(x[k])), mode='constant',
                                                         constant_values=0) for i, x in enumerate(data_holder)],
                                                 axis=0))
                                else:
                                    raise ValueError(
                                        'Unsupported type of attribute use_concat : {}'.format(use_concat[k]))

                    except Exception as e:  # noqa
                        print_yellow("Cannot concat batch data. Perhaps they are of inconsistent shape?")
                        if isinstance(dt, np.ndarray):
                            s = [x[k].shape for x in data_holder]
                            print_yellow("Shape of all arrays to be batched: {}".format(s))
                        try:
                            # open an ipython shell if possible
                            import IPython as IP;
                            IP.embed()  # noqa
                        except ImportError:
                            pass
            else:
                if use_list[k]:
                    result.append(
                        [x[k] for x in data_holder])
                else:
                    dt = data_holder[0][k]
                    if type(dt) in [int, bool]:
                        tp = 'int32'
                    elif type(dt) == float:
                        tp = 'float32'
                    else:
                        try:
                            tp = np.asarray(dt).dtype
                        except AttributeError:
                            raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                    try:
                        result.append(
                            np.asarray([x[k] for x in data_holder], dtype=tp))
                    except Exception as e:  # noqa
                        print_yellow("Cannot batch data. Perhaps they are of inconsistent shape?")
                        if isinstance(dt, np.ndarray):
                            s = [x[k].shape for x in data_holder]
                            print_yellow("Shape of all arrays to be batched: {}".format(s))
                        try:
                            # open an ipython shell if possible
                            import IPython as IP;
                            IP.embed()  # noqa
                        except ImportError:
                            pass

        points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle, label_classes, calib_P, sample_name = result
        feed_dict = dict()
        feed_dict[maps_dict.PL_POINTS_INPUT] = torch.from_numpy(points).float()
        feed_dict[maps_dict.PL_LABEL_SEMSEGS] = torch.from_numpy(sem_labels).long()
        feed_dict[maps_dict.PL_LABEL_DIST] = torch.from_numpy(sem_dists).float()
        feed_dict[maps_dict.PL_LABEL_BOXES_3D] = torch.from_numpy(label_boxes_3d).float()
        feed_dict[maps_dict.PL_LABEL_CLASSES] = torch.from_numpy(label_classes).long()
        feed_dict[maps_dict.PL_ANGLE_CLS] = torch.from_numpy(ry_cls_label).long()
        feed_dict[maps_dict.PL_ANGLE_RESIDUAL] = torch.from_numpy(residual_angle).float()

        feed_dict[maps_dict.PL_CALIB_P2] = torch.from_numpy(calib_P).float()
        feed_dict['sample_name'] = sample_name

        return feed_dict

    # Preprocess data
    def preprocess_samples(self, indices):
        sample_dicts = []
        biggest_label_num = 0
        for sample_idx in indices:
            sample_id = int(self.idx_list[sample_idx])

            img = self.kitti_object.get_image(sample_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_shape = img.shape

            calib = self.kitti_object.get_calibration(sample_id)

            points = self.kitti_object.get_lidar(sample_id)
            points_intensity = points[:, 3:]
            points = points[:, :3]
            # filter out this, first cast it to rect
            points = calib.project_velo_to_rect(points)

            img_points_filter = get_point_filter_in_image(points, calib, image_shape[0], image_shape[1])
            voxelnet_points_filter = get_point_filter(points, self.extents)
            img_points_filter = np.logical_and(img_points_filter, voxelnet_points_filter)
            img_points_filter = np.where(img_points_filter)[0]
            points = points[img_points_filter]
            points_intensity = points_intensity[img_points_filter]

            if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT:
                # then we also need to preprocess groundtruth
                objs = self.kitti_object.get_label_objects(sample_id)
                filtered_obj_list = [obj for obj in objs if obj.type in self.cls_list]
                if len(filtered_obj_list) == 0:
                    # continue if no obj
                    return None, biggest_label_num

                # then is time to generate anchors
                label_boxes_3d = np.array([object_label_to_box_3d(obj) for obj in filtered_obj_list])
                label_boxes_3d = np.reshape(label_boxes_3d, [-1, 7])
                label_classes = np.array([self.cls2idx_dict[obj.type] for obj in filtered_obj_list], np.int)

                # then calculate sem_labels and sem_dists
                tmp_label_boxes_3d = label_boxes_3d.copy()
                # expand by 0.1, so as to cover context information
                tmp_label_boxes_3d[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
                points_mask = check_inside_points(points, tmp_label_boxes_3d)  # [pts_num, gt_num]
                points_cls_index = np.argmax(points_mask, axis=1)  # [pts_num]
                points_cls_index = label_classes[points_cls_index]  # [pts_num]
                sem_labels = np.max(points_mask, axis=1) * points_cls_index  # [pts_num]
                sem_labels = sem_labels.astype(np.int)
                sem_dists = np.ones_like(sem_labels).astype(np.float32)
            else:
                sem_labels = np.ones([points.shape[0]], dtype=np.int)
                sem_dists = np.ones([points.shape[0]], dtype=np.float32)

            points = np.concatenate([points, points_intensity], axis=-1)

            if np.sum(sem_labels) == 0:
                return None, biggest_label_num

            # finally return the sealed result and save as npy file
            if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT:
                sample_dict = {
                    maps_dict.KEY_LABEL_BOXES_3D: label_boxes_3d,
                    maps_dict.KEY_LABEL_CLASSES: label_classes,
                    maps_dict.KEY_LABEL_SEMSEG: sem_labels,
                    maps_dict.KEY_LABEL_DIST: sem_dists,

                    maps_dict.KEY_POINT_CLOUD: points,
                    maps_dict.KEY_STEREO_CALIB: calib,

                    maps_dict.KEY_SAMPLE_NAME: sample_id,
                    maps_dict.KEY_LABEL_NUM: len(label_boxes_3d)
                }
                biggest_label_num = max(len(label_boxes_3d), biggest_label_num)
            else:
                # img_list is test
                sample_dict = {
                    maps_dict.KEY_LABEL_SEMSEG: sem_labels,
                    maps_dict.KEY_LABEL_DIST: sem_dists,
                    maps_dict.KEY_POINT_CLOUD: points,
                    maps_dict.KEY_STEREO_CALIB: calib,
                    maps_dict.KEY_SAMPLE_NAME: sample_id
                }
            sample_dicts.append(sample_dict)
        return sample_dicts, biggest_label_num

    def generate_mixup_sample(self, sample_dict):
        label_boxes_3d = sample_dict[maps_dict.KEY_LABEL_BOXES_3D]
        label_classes = sample_dict[maps_dict.KEY_LABEL_CLASSES]
        points = sample_dict[maps_dict.KEY_POINT_CLOUD]
        label_class_names = np.array([self.idx2cls_dict[label] for label in label_classes])

        tmp_label_boxes_3d = label_boxes_3d.copy()
        # expand by 0.1, so as to cover context information
        tmp_label_boxes_3d[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
        points_mask = check_inside_points(points, tmp_label_boxes_3d)  # [pts_num, gt_num]

        pts_num_inside_box = np.sum(points_mask, axis=0)  # gt_num
        valid_box_idx = np.where(pts_num_inside_box >= cfg.DATASET.MIN_POINTS_NUM)[0]
        if len(valid_box_idx) == 0: return None

        valid_label_boxes_3d = label_boxes_3d[valid_box_idx, :]
        valid_label_classes = label_class_names[valid_box_idx]

        sample_dicts = []
        for index, i in enumerate(valid_box_idx):
            cur_points_mask = points_mask[:, i]
            cur_points_idx = np.where(cur_points_mask)[0]
            cur_inside_points = points[cur_points_idx, :]
            sample_dict = {
                maps_dict.KEY_SAMPLED_GT_POINTS: cur_inside_points,
                maps_dict.KEY_SAMPLED_GT_LABELS_3D: valid_label_boxes_3d[index],
                maps_dict.KEY_SAMPLED_GT_CLSES: valid_label_classes[index],
            }
            sample_dicts.append(sample_dict)
        return sample_dicts

    def preprocess_batch(self):
        # if create_gt_dataset, then also create a boxes_numpy, saving all points
        if cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:  # also save mixup database
            mixup_label_dict = dict([(cls, []) for cls in self.mixup_db_class])
        with open(self.train_list, 'w') as f:
            for i in tqdm.tqdm(range(0, self.sample_num)):
                sample_dicts, tmp_biggest_label_num = self.preprocess_samples([i])
                if sample_dicts is None:
                    # print('%s has no ground truth or ground truth points'%self.idx_list[i].name)
                    continue
                # else save the result
                f.write('%06d.npy\n' % i)
                np.save(os.path.join(self.sv_npy_path, '%06d.npy' % i), sample_dicts[0])

                # create_gt_dataset
                if self.img_list in ['train', 'val',
                                     'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
                    # then also parse the sample_dicts so as to generate mixup database
                    mixup_sample_dicts = self.generate_mixup_sample(sample_dicts[0])
                    if mixup_sample_dicts is None: continue
                    for mixup_sample_dict in mixup_sample_dicts:
                        cur_cls = mixup_sample_dict[maps_dict.KEY_SAMPLED_GT_CLSES]
                        mixup_label_dict[cur_cls].append(mixup_sample_dict)

        if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
            print('**** Generating groundtruth database ****')
            for cur_cls_name, mixup_sample_dict in mixup_label_dict.items():
                cur_mixup_db_cls_path = self.mixup_db_cls_path[cur_cls_name]
                cur_mixup_db_trainlist_path = self.mixup_db_trainlist_path[cur_cls_name]
                print('**** Class %s ****' % cur_cls_name)
                with open(cur_mixup_db_trainlist_path, 'w') as f:
                    for tmp_idx, tmp_cur_mixup_sample_dict in tqdm.tqdm(enumerate(mixup_sample_dict)):
                        f.write('%06d.npy\n' % tmp_idx)
                        np.save(os.path.join(cur_mixup_db_cls_path, '%06d.npy' % tmp_idx), tmp_cur_mixup_sample_dict)
        print('Ending of the preprocess !!!')

    # Evaluation

    def generate_annotations(self, input_dict, pred_dicts, class_names, cls_thresh=0.0005, save_to_file=False, output_dir=None):
        def get_empty_prediction():
            ret_dict = {
                'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([]),
                'boxes_lidar': np.zeros([0, 7])
            }
            return ret_dict

        def generate_single_anno(idx, input_dict, pred_dict, class_names):
            num_example = 0

            if len(pred_dict['pred_3d_bbox'][idx]) == 0:
                return get_empty_prediction(), num_example

            # area_limit = image_shape = None
            # if cfg.MODEL.TEST.BOX_FILTER['USE_IMAGE_AREA_FILTER']:
            #     image_shape = input_dict['image_shape'][idx]
            #     area_limit = image_shape[0] * image_shape[1] * 0.8

            sample_name = input_dict['sample_name'][idx]
            calib_P = input_dict['frame_calib_p2'][idx].cpu().numpy()

            pred_cls_score = pred_dict['pred_3d_score'][idx].detach().cpu().numpy()
            pred_bbox_3d = pred_dict['pred_3d_bbox'][idx].detach().cpu().numpy()
            pred_cls_category = pred_dict['pred_3d_class_Category'][idx].detach().cpu().numpy()

            select_idx = np.where(pred_cls_score >= cls_thresh)[0]
            if len(select_idx) == 0:
                return get_empty_prediction(), num_example
            pred_cls_score = pred_cls_score[select_idx]
            pred_cls_category = pred_cls_category[select_idx]
            pred_bbox_3d = pred_bbox_3d[select_idx]

            pred_bbox_corners = get_box3d_corners_helper_np(pred_bbox_3d[:, :3], pred_bbox_3d[:, -1], pred_bbox_3d[:, 3:-1])
            pred_bbox_2d = project_to_image_space_corners(pred_bbox_corners, calib_P)

            obj_num = len(pred_bbox_3d)
            obj_detection = np.zeros([obj_num, 14], np.float32)
            if 'Car' not in self.cls_list:
                pred_cls_category += 1
            obj_detection[:, 0] = pred_cls_category
            obj_detection[:, 1:5] = pred_bbox_2d
            obj_detection[:, 6:9] = pred_bbox_3d[:, :3]
            obj_detection[:, 9] = pred_bbox_3d[:, 4]  # h
            obj_detection[:, 10] = pred_bbox_3d[:, 5]  # w
            obj_detection[:, 11] = pred_bbox_3d[:, 3]  # l
            obj_detection[:, 12] = pred_bbox_3d[:, 6]  # ry
            obj_detection[:, 13] = pred_cls_score

            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': []}

            for i in range(obj_num):
                anno['name'].append(class_names[int(pred_cls_category[i] - 1)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(pred_bbox_2d[i])
                anno['dimensions'].append(pred_bbox_3d[i][3:6])  # lwh
                anno['location'].append(pred_bbox_3d[i][:3])
                anno['rotation_y'].append(pred_bbox_3d[i][6])
                anno['score'].append(pred_cls_score[i])

                num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = get_empty_prediction()

            return anno, num_example

        annos = []
        for i in range(len(pred_dicts['pred_3d_bbox'])):
            sample_idx = input_dict['sample_name'][i]
            single_anno, num_example = generate_single_anno(i, input_dict, pred_dicts, self.cls_list)
            single_anno['num_example'] = num_example
            single_anno['sample_idx'] = np.array([sample_idx] * num_example, dtype=np.int64)
            annos.append(single_anno)
            if save_to_file:
                cur_det_file = os.path.join(output_dir, '%s.txt' % sample_idx)
                with open(cur_det_file, 'w') as f:
                    bbox = single_anno['bbox']
                    loc = single_anno['location']
                    dims = single_anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_anno['name'][idx], single_anno['alpha'][idx], bbox[idx][0], bbox[idx][1],
                                 bbox[idx][2], bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_anno['rotation_y'][idx], single_anno['score'][idx]),
                              file=f)

        gt_annos = []
        for idx in range(len(input_dict['label_boxes_3d_pl'])):
            gt_3d_bbox = input_dict['label_boxes_3d_pl'][idx].cpu().numpy()
            select_idx = np.any(gt_3d_bbox != 0, axis=-1)
            gt_3d_bbox = gt_3d_bbox[select_idx]

            gt_label = input_dict['label_classes_pl'][idx].cpu().numpy()
            gt_label = gt_label[select_idx]

            if len(gt_3d_bbox) == 0:
                gt_annos.append(get_empty_prediction())
                continue

            calib_P = input_dict['frame_calib_p2'][idx].cpu().numpy()

            gt_bbox_corners = get_box3d_corners_helper_np(gt_3d_bbox[:, :3], gt_3d_bbox[:, -1], gt_3d_bbox[:, 3:-1])
            gt_bbox_2d = project_to_image_space_corners(gt_bbox_corners, calib_P)

            obj_num = len(gt_3d_bbox)
            obj_detection = np.zeros([obj_num, 14], np.float32)
            # if 'Car' not in self.cls_list:
            #     pred_cls_category += 1
            obj_detection[:, 0] = gt_label
            obj_detection[:, 1:5] = gt_bbox_2d
            obj_detection[:, 6:9] = gt_3d_bbox[:, :3]
            obj_detection[:, 9] = gt_3d_bbox[:, 4]  # h
            obj_detection[:, 10] = gt_3d_bbox[:, 5]  # w
            obj_detection[:, 11] = gt_3d_bbox[:, 3]  # l
            obj_detection[:, 12] = gt_3d_bbox[:, 6]  # ry

            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': []}

            for i in range(obj_num):
                anno['name'].append(class_names[int(gt_label[i] - 1)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(gt_bbox_2d[i])
                anno['dimensions'].append(gt_3d_bbox[i][3:6])  # lwh
                anno['location'].append(gt_3d_bbox[i][:3])
                anno['rotation_y'].append(gt_3d_bbox[i][6])

            anno = {k: np.stack(v) for k, v in anno.items()}

            gt_annos.append(anno)

        assert len(annos) == len(gt_annos)

        return annos, gt_annos

    def evaluation(self, det_annos, gt_annos):
        import lib.dataset.kitti_object_eval_python.eval as kitti_eval
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = copy.deepcopy(gt_annos)
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, self.cls_list)

        return ap_result_str, ap_dict


