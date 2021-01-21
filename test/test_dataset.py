import warnings
warnings.filterwarnings('ignore')

from lib.core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from lib.core.config import cfg
from lib.modeling import choose_model
from lib.dataset.dataloader import choose_dataset
from torch.utils.data import DataLoader
from lib.builder.loss_builder import LossBuilder
import numpy as np
import torch
from lib.builder.target_assigner import TargetAssigner
# Init datasets and dataloaders

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

cfg_from_file("../configs/kitti/3dssd/3dssd.yaml")
dataset_func = choose_dataset()
dataset = dataset_func('loading', split="training", img_list="train", is_training=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=6, worker_init_fn=my_worker_init_fn,  collate_fn=dataset.load_batch)
assigner = TargetAssigner(0)
model_func = choose_model()
model = model_func(1, is_training=True).cuda()
if __name__ == '__main__':
    for batch_idx, batch_data_label in enumerate(dataloader):
        for key in batch_data_label:
            if isinstance(batch_data_label[key], torch.Tensor):
                batch_data_label[key] = batch_data_label[key].cuda()
        for k, v in batch_data_label.items():
            print(k, v.shape)
        
        #returned_list = assigner.assign(batch_data_label['point_cloud_pl'][..., :3],
        #                                torch.unsqueeze((batch_data_label['point_cloud_pl'][..., :3]), dim=2),
        #                                batch_data_label['label_boxes_3d_pl'],
        #                                batch_data_label['label_classes_pl'],
        #                                batch_data_label['angle_cls_pl'],
        #                                batch_data_label['angle_res_pl'])
        #assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute = returned_list
        #print(assigned_idx.shape)
        end_points = model(batch_data_label)
        loss_builder = LossBuilder(0)
        index = -1
        total_loss, loss_dict = loss_builder.compute_loss(index, end_points, corner_loss_flag=True, vote_loss_flag=True)
        print(loss_dict)
        break
