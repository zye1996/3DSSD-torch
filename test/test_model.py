from lib.core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from lib.core.config import cfg
from lib.modeling import choose_model

if __name__ == '__main__':
    cfg_from_file("../configs/kitti/3dssd/3dssd.yaml")
    model_func = choose_model()
    model = model_func(1, is_training=True)
    print(model)