from lib.core.config import cfg
import torch


class LRScheduler(object):
    def __init__(self, optimizer):
        steps = cfg.SOLVER.STEPS
        self.steps = steps
        self.values = [cfg.SOLVER.BASE_LR] + [cfg.SOLVER.BASE_LR * cfg.SOLVER.GAMMA ** (index + 1) for index, step in
                                              enumerate(steps)]
        self.optimizer = optimizer

    def step(self, it):
        index = -1
        for i in range(len(self.steps)):
            if i < self.steps[i]:
                index = i

        new_lr = self.values[index]
        self.optimizer.lr = new_lr


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
