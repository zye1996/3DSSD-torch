#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import rospy
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge
from inference import *
from kitti_util import *
from parseTrackletXML import *
from pub_utils import *
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray


def read_object(tracklet_file, calib_file):
    calib = Calibration(calib_file, from_video=True)
    tracklets = parseXML(trackletFile=tracklet_file)
    frame_dict_2d = defaultdict(list)
    frame_dict_3d = defaultdict(list)
    frame_dict_obj= defaultdict(list)
    frame_dict_id = defaultdict(list)
    for iTracklet, tracklet in enumerate(tracklets):
        # print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        h,w,l = tracklet.size
        trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
                in tracklet:
            #if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
            #    continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]   # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([\
              [np.cos(yaw), -np.sin(yaw), 0.0], \
              [np.sin(yaw),  np.cos(yaw), 0.0], \
              [        0.0,          0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

            # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
            #   car-centered yaw (i.e. 0 degree = same orientation as car).
            #   makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yawVisual = ( yaw - np.arctan2(y, x) ) % (2.*np.pi)
            frame_dict_3d[absoluteFrameNumber].append(cornerPosInVelo.T)
            cornerPosInImg = calib.project_velo_to_image(cornerPosInVelo.T)
            frame_dict_2d[absoluteFrameNumber].append([np.min(cornerPosInImg, axis=0)[0], np.min(cornerPosInImg, axis=0)[1],
                                                        np.max(cornerPosInImg, axis=0)[0], np.max(cornerPosInImg, axis=0)[1]])
            frame_dict_obj[absoluteFrameNumber].append(tracklet.objectType)
            frame_dict_id[absoluteFrameNumber].append(iTracklet)
    return frame_dict_2d, frame_dict_3d, frame_dict_obj, frame_dict_id
        

if __name__ == "__main__":

    args = rospy.myargv(argv=sys.argv)
    if len(args) != 4:
        print("ERROR: args not provided")
        sys.exit(1)

    CONF_PATH = args[1]  # '/home/yzy/PycharmProjects/3DSSD-torch/tools/cfgs/kitti_models/3dssd.yaml'
    CKPT_PATH = args[2]  # '/home/yzy/PycharmProjects/3DSSD-torch/output/kitti_models/3dssd/default/ckpt/checkpoint_epoch_120.pth'
    DATA_ROOT = args[3] #'/home/yzy/Documents/dataset/kitti/raw/2011_09_26/'
    DATA_PATH = [os.path.join(DATA_ROOT, x) for x in os.listdir(DATA_ROOT) if x.startswith(os.path.split(DATA_ROOT)[-1])][-1]



    # load model config
    cfg_from_yaml_file(CONF_PATH, cfg)
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(os.path.join(DATA_PATH, "velodyne_points/data")),
        ext=".bin")
    detector = Detector(cfg,
                       ckpt_file=CKPT_PATH,
                       calib_file=DATA_ROOT,
                       demo_dataset=demo_dataset)

    rospy.init_node('kitti_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_pcl', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego', MarkerArray, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3dboxes', MarkerArray, queue_size=100)
    box3d_pred_pub = rospy.Publisher('kitti_pred_3dboxes', MarkerArray, queue_size=100)
    bridge = CvBridge()

    frame_dict_2d, frame_dict_3d, frame_dict_obj, frame_dict_id = read_object(os.path.join(DATA_PATH, "tracklet_labels.xml"), DATA_ROOT)

    rate = rospy.Rate(10)
    frame = 0

    while not rospy.is_shutdown():

        pc_path = os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin' % frame)
        img_path = os.path.join(DATA_PATH, 'image_02/data/%010d.png' % frame)


        # load data
        data_dict = demo_dataset[frame]
        img = cv2.imread(img_path)
        point_cloud = data_dict['raw_points']

        # run inference
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_2d, pred_3d = detector.run(data_dict, threshold=0.2)
        frame_dict_2d_pred, frame_dict_3d_pred = detector.run(data_dict)


        # publish data
        publish_point_cloud(pcl_pub, point_cloud)
        publish_camera(cam_pub, bridge, img, frame_dict_2d[frame] + frame_dict_2d_pred,
                       frame_dict_obj[frame]+['Prediction' for i in range(len(frame_dict_3d_pred))])
        publish_3dbox(box3d_pub, frame_dict_3d[frame], frame_dict_id[frame], frame_dict_obj[frame], publish_id=False, publish_distance=False)
        publish_3dbox(box3d_pred_pub, frame_dict_3d_pred,
                      [-i for i in range(1, len(frame_dict_3d_pred)+1)],
                      ['Prediction' for i in range(len(frame_dict_3d_pred))],
                      publish_id=False, publish_distance=False)
        publish_ego_car(ego_pub)
        rospy.loginfo('camera image published')
        rate.sleep()
        frame += 1
        frame %= len(demo_dataset)
