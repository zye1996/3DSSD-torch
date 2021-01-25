# from .double_stage_detector import DoubleStageDetector
from lib.core.config import cfg

from .single_stage_detector import SingleStageDetector


def choose_model():
    model_dict = {
        'SingleStage': SingleStageDetector,
        # 'DoubleStage': DoubleStageDetector,
    } 

    return model_dict[cfg.MODEL.TYPE]
