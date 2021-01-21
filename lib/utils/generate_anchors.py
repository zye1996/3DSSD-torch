import numpy as np
import torch

from lib.core.config import cfg

def generate_3d_anchors_by_point(points, anchor_sizes):
    '''
        Generate 3d anchors by points
        points: [b, n, 3]
        anchors_size: [cls_num, 3], l, h, w

        Return [b, n, cls_num, 7]
    '''
    bs, points_num, _ = points.shape
    anchor_sizes = np.array(anchor_sizes) # [cls_num, 3]
    anchors_num = len(anchor_sizes)
   
    # then generate anchors for each points 
    ctr = np.tile(np.reshape(points, [bs, points_num, 1, 3]), [1, 1, anchors_num, 1]) # [x, y, z]

    offset = np.tile(np.reshape(anchor_sizes, [1, 1, anchors_num, 3]), [bs, points_num, 1, 1]) # [l, h, w]

    # then sub y to force anchors on the center
    ctr[:, :, :, 1] += offset[:, :, :, 1] / 2.
    ry = np.zeros([bs, points_num, anchors_num, 1], dtype=ctr.dtype)
    
    all_anchor_boxes_3d = np.concatenate([ctr, offset, ry], axis=-1)
    all_anchor_boxes_3d = np.reshape(all_anchor_boxes_3d, [bs, points_num, anchors_num, 7])

    return all_anchor_boxes_3d


def generate_3d_anchors_by_point_torch(points, anchor_sizes):
    '''
        Generate 3d anchors by points
        points: [bs, n, 3], xyz, the location of this points
        anchor_sizes: [cls_num, 3], lhw
    
        Return [b, n, cls_num, 7]
    '''
    bs, points_num, _ = points.get_shape().as_list()

    anchor_sizes = np.array(anchor_sizes).astype(np.float32)
    anchors_num = len(anchor_sizes)
    
    # then generate anchors for each points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]  # [bs, points_num]
    x = x.view(bs, points_num, 1, 1).repeat(1, 1, anchors_num, 1)
    y = y.view(bs, points_num, 1, 1).repeat(1, 1, anchors_num, 1)
    z = z.view(bs, points_num, 1, 1).repeat(1, 1, anchors_num, 1)

    # then sub y_ctr by the anchor_size
    l, h, w = anchor_sizes[:, 0], anchor_sizes[:, 1], anchor_sizes[:, 2] # [anchors_num]
    l = l.view(bs, points_num, 1, 1).repeat(1, 1, anchors_num, 1)
    h = h.view(bs, points_num, 1, 1).repeat(1, 1, anchors_num, 1)
    w = w.view(bs, points_num, 1, 1).repeat(1, 1, anchors_num, 1)

    ry = torch.zeros_like(l) # [bs, points_num, anchors_num]

    y = y + h / 2.

    all_anchor_boxes_3d = torch.concat([x, y, z, l, h, w, ry], axis=-1)
    all_anchor_boxes_3d = all_anchor_boxes_3d.view(bs, points_num, anchors_num, 7).to(points.device)
    return all_anchor_boxes_3d 
