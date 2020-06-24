import logging
import os
import torch
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0/batch_size))
    return res


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def save_model(filename, model, optimizer, scheduler, epoch, **kwargs):
    # Torch Save State Dict
    state = {
        'epoch': epoch+1,
        'model': model.state_dict() if model is not None else None,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, filename+'.pth')
    return


def load_model(file_name, model, optimizer, scheduler, **kwargs):
    checkpoints = torch.load(file_name + '.pth')
    if model is not None and checkpoints['model'] is not None:
        model.load_state_dict(checkpoints['model'])
    if optimizer is not None and checkpoints['optimizer'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer'])
    if scheduler is not None and checkpoints['scheduler'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler'])
    print("%s Loaded!" % (file_name))
    return checkpoints


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6
