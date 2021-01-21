import os, sys
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
import datetime
import time
from tensorboardX import SummaryWriter
import tqdm
from torch.nn.utils import clip_grad_norm_

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from lib.dataset.dataloader import choose_dataset

from lib.modeling import choose_model
from lib.core.trainer_utils import LRScheduler, save_checkpoint, checkpoint_state
from lib.utils.common_util import create_logger
from lib.modeling.single_stage_detector import post_process


def parse_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--cfg', required=True, help='Config file for training')
    parser.add_argument('--restore_model_path', required=True, help='Restore model path e.g. log/model.ckpt [default: None]')
    parser.add_argument('--img_list', default='val', help='Train/Val/Trainval list')
    parser.add_argument('--split', default='training', help='Dataset split')

    # some evaluation threshold
    parser.add_argument('--cls_threshold', default=0, help='Filtering Predictions')
    parser.add_argument('--eval_interval_secs', default=300, help='Sleep after one evaluation loop')
    parser.add_argument('--no_gt', action='store_true', help='Used for test set')
    args = parser.parse_args()
    return args


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Evaluator:
    def __init__(self, args):
        self.batch_size = cfg.TRAIN.CONFIG.BATCH_SIZE
        self.gpu_num = cfg.TRAIN.CONFIG.GPU_NUM
        self.num_workers = cfg.DATA_LOADER.NUM_THREADS
        self.log_dir = cfg.MODEL.PATH.EVALUATION_DIR
        self.is_training = False

        self.cls_thresh = float(args.cls_threshold)
        self.eval_interval_secs = args.eval_interval_secs
        self.restore_model_path = args.restore_model_path

        # save dir
        self.log_dir = args.restore_model_path[0:args.restore_model_path.find('/ckpt')]
        self.logger = create_logger(os.path.join(self.log_dir, 'log_eval.txt'))
        self.logger.info(str(args) + '\n')
        self.result_dir = os.path.join(self.log_dir, 'eval')
        self.logger.info('**** Saving Evaluation results to the path %s ****' % self.result_dir)

        # dataset
        dataset_func = choose_dataset()
        self.dataset = dataset_func('loading', split=args.split, img_list=args.img_list, is_training=self.is_training)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size * self.gpu_num, shuffle=False,
                                     num_workers=self.num_workers, worker_init_fn=my_worker_init_fn,
                                     collate_fn=self.dataset.load_batch)

        self.logger.info('**** Dataset length is %d ****' % len(self.dataset))
        self.val_size = len(self.dataset)

        # model
        self.model_func = choose_model()
        self.model = self.model_func(self.batch_size, self.is_training)
        self.model = self.model.cuda()

        # tensorboard
        self.tb_log = SummaryWriter(log_dir=os.path.join(self.result_dir, 'tensorboard'))


    def evaluate(self):
        if os.path.isdir(self.restore_model_path):
            self.evaluate_all_ckpts()
        else:
            self.evaluate_one_ckpts()

    def evaluate_all_ckpts(self):
        pass

    def evaluate_one_ckpts(self, ckpt_file=None):
        if ckpt_file is None:
            ckpt_file = self.restore_model_path

        # load from checkpoint
        if args.restore_model_path is not None:
            self.model.load_params_from_file(filename=ckpt_file, logger=self.logger)

        if self.gpu_num > 1:
            self.logger.info("Use %d GPUs!" % self.gpu_num)
            self.model = torch.nn.DataParallel(self.model)

        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='eval', dynamic_ncols=True)
        start_time = time.time()

        det_annos = []
        gt_annos = []

        for batch_idx, batch_data_label in enumerate(self.dataloader):

            for key in batch_data_label:
                if isinstance(batch_data_label[key], torch.Tensor):
                    batch_data_label[key] = batch_data_label[key].cuda()

            self.model.eval()
            end_points = self.model(batch_data_label)
            end_points = post_process(end_points)

            det_anno, gt_anno = self.dataset.generate_annotations(batch_data_label, end_points, self.dataset.cls_list, save_to_file=False, output_dir=None)
            det_annos = det_annos + det_anno
            gt_annos = gt_annos + gt_anno

            # progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        progress_bar.close()

        self.logger.info('*************** Performance of %s *****************' % ckpt_file)
        result_str, result_dict = self.dataset.evaluation(det_annos, gt_annos)

        self.logger.info(result_str)
        self.logger.info('****************Evaluation done.*****************')





if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    # set bs, gpu_num and workers_num to be 1
    # cfg.TRAIN.CONFIG.BATCH_SIZE = 1 # only support bs=1 when testing
    # cfg.TRAIN.CONFIG.GPU_NUM = 1
    # cfg.DATA_LOADER.NUM_THREADS = 1

    if args.no_gt:
        cfg.TEST.WITH_GT = False

    cur_evaluator = Evaluator(args)
    cur_evaluator.evaluate()
    print("**** Finish evaluation steps ****")