from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from ..train_utils.optimization import build_optimizer, build_scheduler
import os

os.chdir("../tools")

cfg_from_yaml_file("../tools/cfgs/kitti_models/pointrcnn_iou.yaml", cfg)
print(cfg['OPTIMIZATION'])