import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import tqdm
from pcdet.utils.calibration_kitti import Calibration, get_calib_from_file
from pcdet.utils.object3d_kitti import get_objects_from_label
from PIL import Image
from torchvision import transforms


class Painter:

    def __init__(self):
        self.root_split_path = "../../../data/kitti/training/"
        self.save_path = "/home/yzy/PycharmProjects/OpenPCDet/data/kitti/training/velodyne_painted_mono/"
        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet101', pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, left):
        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        output_permute = output.permute(1, 2, 0)
        output_probability, output_predictions = output_permute.max(2)

        other_object_mask = ~((output_predictions == 0) | (output_predictions == 2) | (output_predictions == 7) | (
                    output_predictions == 15))
        detect_object_mask = ~other_object_mask
        sf = torch.nn.Softmax(dim=2)

        # bicycle = 2  car = 7 person = 15 background = 0
        output_reassign = torch.zeros(output_permute.size(0), output_permute.size(1), 4)
        output_reassign[:, :, 0] = detect_object_mask * output_permute[:, :,
                                                        0] + other_object_mask * output_probability  # background
        output_reassign[:, :, 1] = output_permute[:, :, 2]  # bicycle
        output_reassign[:, :, 2] = output_permute[:, :, 7]  # car
        output_reassign[:, :, 3] = output_permute[:, :, 15]  # person
        output_reassign_softmax = sf(output_reassign)

        return output_reassign_softmax, np.array(input_image)

    def get_label(self, idx):
        label_file = self.root_split_path + 'label_2/' + ('%s.txt' % idx)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        return Calibration(calib_file)

    def get_calib_fromfile(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        calib = get_calib_from_file(calib_file)
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'], np.array([[0., 0., 0., 1.]])], axis=0)
        return calib

    def create_cyclist(self, augmented_lidar):
        bike_idx = np.where(augmented_lidar[:, 5] >= 0.2)[0]  # 0, 1(bike), 2, 3(person)
        bike_points = augmented_lidar[bike_idx]
        cyclist_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
        for i in range(bike_idx.shape[0]):
            cyclist_mask = (np.linalg.norm(augmented_lidar[:, :3] - bike_points[i, :3], axis=1) < 1) & (
                        np.argmax(augmented_lidar[:, -4:], axis=1) == 3)
            if np.sum(cyclist_mask) > 0:
                cyclist_mask_total |= cyclist_mask
            else:
                augmented_lidar[bike_idx[i], 4], augmented_lidar[bike_idx[i], 5] = augmented_lidar[bike_idx[i], 5], 0
        augmented_lidar[cyclist_mask_total, 7], augmented_lidar[cyclist_mask_total, 5] = 0, augmented_lidar[
            cyclist_mask_total, 7]
        return augmented_lidar

    def cam_to_lidar(self, pointcloud, projection_mats):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords
        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1])  # copy reflectances column
        lidar_velo_coords[:, -1] = 1  # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo2cam'].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances

        return lidar_cam_coords

    def augment_lidar_class_scores(self, class_scores, lidar_raw, projection_mats, image_rgb=None):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection
        points_projected_on_mask = projection_mats['P2'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask = points_projected_on_mask.transpose()
        points_projected_on_mask = points_projected_on_mask / (points_projected_on_mask[:, 2].reshape(-1, 1))

        true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (
                    points_projected_on_mask[:, 0] < class_scores.shape[1])  # x in img coords is cols of img
        true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (
                    points_projected_on_mask[:, 1] < class_scores.shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

        points_projected_on_mask = points_projected_on_mask[
            true_where_point_on_img]  # filter out points that don't project to image
        lidar_cam_coords = lidar_cam_coords[true_where_point_on_img]
        points_projected_on_mask = np.floor(points_projected_on_mask).astype(int)  # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask = points_projected_on_mask[:, :2]  # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        # indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        # socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores = class_scores[points_projected_on_mask[:, 1], points_projected_on_mask[:, 0]].reshape(-1, class_scores.shape[2])
        if image_rgb is not None:
            point_colors = (image_rgb[points_projected_on_mask[:, 1], points_projected_on_mask[:, 0]].reshape(-1,
                                                                                                             3).astype(
                np.float32) - 128.0) / 255
            augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores, point_colors), axis=1)
        else:
            augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = self.create_cyclist(augmented_lidar)

        # todo fix here
        augmented_lidar = augmented_lidar[..., [0, 1, 2, 3, 4, 6, 7, 5]]
        return augmented_lidar

    def augment_lidar_class_scores_both(self, class_scores_r, class_scores_l, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        # lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats?
        ################################
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)

        # right
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_r = projection_mats['P3'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        points_projected_on_mask_r = points_projected_on_mask_r / (points_projected_on_mask_r[:, 2].reshape(-1, 1))

        true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (
                    points_projected_on_mask_r[:, 0] < class_scores_r.shape[1])  # x in img coords is cols of img
        true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (
                    points_projected_on_mask_r[:, 1] < class_scores_r.shape[0])
        true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r

        points_projected_on_mask_r = points_projected_on_mask_r[
            true_where_point_on_img_r]  # filter out points that don't project to image
        points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(
            int)  # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_r = points_projected_on_mask_r[:,
                                     :2]  # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        # left
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_l = projection_mats['P2'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l / (points_projected_on_mask_l[:, 2].reshape(-1, 1))

        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (
                    points_projected_on_mask_l[:, 0] < class_scores_l.shape[1])  # x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (
                    points_projected_on_mask_l[:, 1] < class_scores_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[
            true_where_point_on_img_l]  # filter out points that don't project to image
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(
            int)  # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_l = points_projected_on_mask_l[:,
                                     :2]  # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r

        # indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        # socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_r = class_scores_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1,
                                                                                                                    class_scores_r.shape[
                                                                                                                        2])
        point_scores_l = class_scores_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1,
                                                                                                                    class_scores_l.shape[
                                                                                                                        2])
        # augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores_r.shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_r, -class_scores_r.shape[2]:] += point_scores_r
        augmented_lidar[true_where_point_on_img_l, -class_scores_l.shape[2]:] += point_scores_l
        augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:] = 0.5 * augmented_lidar[
                                                                                         true_where_point_on_both_img,
                                                                                         -class_scores_r.shape[2]:]
        augmented_lidar = augmented_lidar[true_where_point_on_img]
        augmented_lidar = self.create_cyclist(augmented_lidar)

        return augmented_lidar

    def run(self):
        num_image = 7481
        for idx in tqdm.trange(num_image):
            sample_idx = "%06d" % idx
            points = self.get_lidar(sample_idx)

            # add socres here
            scores_from_cam, img_l = self.get_score(sample_idx, "image_2/")
            # scores_from_cam_r, img_r = self.get_score(sample_idx, "image_3/")
            # points become [points, scores]
            calib_info = self.get_calib_fromfile(sample_idx)

            points = self.augment_lidar_class_scores(scores_from_cam, points, calib_info, None)
            #points = self.augment_lidar_class_scores_both(scores_from_cam_r.cpu().numpy(),
            #                                              scores_from_cam.cpu().numpy(), points, calib_info)
            np.save(self.save_path + ("%06d.npy" % idx), points)


if __name__ == '__main__':
    painter = Painter()
    painter.run()
