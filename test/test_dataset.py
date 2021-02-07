from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

cfg_file = "../tools/cfgs/kitti_models/ssd3d.yaml"
cfg_from_yaml_file(cfg_file, cfg)

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
print(data.keys())