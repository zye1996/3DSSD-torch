from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2MSG_FPS
from pcdet.models.backbones_3d.pfe import VoteModule
from pcdet.models.dense_heads.point_head_box_3dssd import PointHeadBox3DSSD
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
import torch
import numpy as np

cfg_file = "../tools/cfgs/kitti_models/pointrcnn_bb_replaced.yaml"
cfg_from_yaml_file(cfg_file, cfg)
print(cfg['MODEL']["BACKBONE_3D"])
cfg_backbone = cfg['MODEL']["BACKBONE_3D"]
cfg_backbone.pop('NAME')

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


logger = common_utils.create_logger('test.txt', rank=cfg.LOCAL_RANK)

train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=2,
    dist=False, workers=6,
    logger=logger,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=1
)

data = next(iter(train_loader))
load_data_to_gpu(data)

points = torch.rand(16384*2, 4).cuda()
batch_idx = torch.zeros(16384*2, 1).cuda()
batch_idx[16384:, 0] = 1

dummy_input = dict(
    batch_size=2,
    points=torch.cat([batch_idx, points], dim=1)
)
backbone = PointNet2MSG_FPS(cfg_backbone, input_channels=4).cuda()
print(backbone)
result = backbone(data)

for k, v in result.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

'''
cfg_pfe = cfg['MODEL']["PFE"]
cfg_pfe.pop('NAME')

for k, v in result.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

pfe = VoteModule(cfg_pfe, None, None).cuda()
print(pfe)
pfe_result = pfe(result)

for k, v in pfe_result.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

cfg_head = cfg['MODEL']["POINT_HEAD"]
cfg_head.pop("NAME")
head = PointHeadBox3DSSD(1, 512, cfg_head).cuda()

head_result = head(pfe_result)
print(head_result)
'''