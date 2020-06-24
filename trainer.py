import time
import torch
import os
from util import log_display, accuracy, AverageMeter

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer():
    def __init__(self, data_loader, logger, config, name='Trainer', metrics='classfication'):
        self.data_loader = data_loader
        self.logger = logger
        self.name = name
        self.step = 0
        self.config = config
        self.log_frequency = config.log_frequency
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()
        self.report_metrics = self.classfication_metrics if metrics == 'classfication' else self.regression_metrics

    def train(self, epoch, GLOBAL_STEP, model, optimizer, criterion):
        model.train()
        for images, labels in self.data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            self.train_batch(images, labels, model, criterion, optimizer)
            self.log(epoch, GLOBAL_STEP)
            GLOBAL_STEP += 1
        return GLOBAL_STEP

    def train_batch(self, x, y, model, criterion, optimizer):
        start = time.time()
        model.zero_grad()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_bound)
        optimizer.step()
        self.report_metrics(pred, y, loss)
        self.logger_payload['lr'] = optimizer.param_groups[0]['lr'],
        self.logger_payload['|gn|'] = grad_norm
        end = time.time()
        self.step += 1
        self.time_used = end - start

    def log(self, epoch, GLOBAL_STEP):
        if GLOBAL_STEP % self.log_frequency == 0:
            display = log_display(epoch=epoch,
                                  global_step=GLOBAL_STEP,
                                  time_elapse=self.time_used,
                                  **self.logger_payload)
            self.logger.info(display)

    def classfication_metrics(self, x, y, loss):
        acc, acc5 = accuracy(x, y, topk=(1, 5))
        self.loss_meters.update(loss.item(), y.shape[0])
        self.acc_meters.update(acc.item(), y.shape[0])
        self.acc5_meters.update(acc5.item(), y.shape[0])
        self.logger_payload = {"acc": acc,
                               "acc_avg": self.acc_meters.avg,
                               "loss": loss,
                               "loss_avg": self.loss_meters.avg}

    def regression_metrics(self, x, y, loss):
        diff = abs((x - y).mean().detach().item())
        self.loss_meters.update(loss.item(), y.shape[0])
        self.acc_meters.update(diff, y.shape[0])
        self.logger_payload = {"|diff|": diff,
                               "|diff_avg|": self.acc_meters.avg,
                               "loss": loss,
                               "loss_avg": self.loss_meters.avg}

    def _reset_stats(self):
        self.loss_meters.reset()
        self.acc_meters.reset()
        self.acc5_meters.reset()
