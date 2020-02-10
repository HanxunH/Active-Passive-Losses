import argparse
import torch
import time
import os
import collections
import pickle
import logging
import torchvision
from tqdm import tqdm
from model import SCEModel, ResNet34
from dataset import DatasetGenerator, Clothing1MDatasetLoader, ImageNetDatasetLoader
from utils.utils import AverageMeter, accuracy, count_parameters_in_MB
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from train_util import TrainUtil
from loss import *

# ArgParse
parser = argparse.ArgumentParser(description='RobustLoss')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--l2_reg', type=float, default=1e-4)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--train_log_every', type=int, default=100)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', default='data', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints/cifar10/', type=str)
parser.add_argument('--data_nums_workers', type=int, default=4)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--nr', type=float, default=0.4, help='noise_rate')
parser.add_argument('--loss', type=str, default='SCE', help='SCE, CE, NCE, MAE, RCE')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha scale')
parser.add_argument('--beta', type=float, default=1.0, help='beta scale')
parser.add_argument('--q', type=float, default=0.7, help='q for gce')
parser.add_argument('--gamma', type=float, default=2, help='gamma for FocalLoss')
parser.add_argument('--dataset_type', choices=['mnist', 'cifar10', 'cifar100', 'clothing1m', 'imagenet'], type=str, default='cifar10')
parser.add_argument('--scale_exp', action='store_true', default=False)
parser.add_argument('--alpha_beta_exp', action='store_true', default=False)
parser.add_argument('--version', type=str, default='robust_loss')
parser.add_argument('--run_version', type=str, default='run1')
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

if args.dataset_type == 'cifar100':
    args.checkpoint_path = 'checkpoints/cifar100/'
    log_dataset_type = 'cifar100'
elif args.dataset_type == 'cifar10':
    args.checkpoint_path = 'checkpoints/cifar10/'
    log_dataset_type = 'cifar10'
elif args.dataset_type == 'mnist':
    args.checkpoint_path = 'checkpoints/mnist/'
    log_dataset_type = 'mnist'
elif args.dataset_type == 'clothing1m':
    args.checkpoint_path = 'checkpoints/clothing1m/'
    log_dataset_type = 'clothing1m'
elif args.dataset_type == 'imagenet':
    args.checkpoint_path = 'checkpoints/ILSVR2012/'
    log_dataset_type = 'imagenet'
else:
    raise('Unknown Dataset')

log_sym_type = ''
if args.dataset_type == 'clothing1m':
    log_dataset_type = 'clothing1m'
elif args.dataset_type == 'imagenet':
    log_dataset_type = 'imagenet'
elif not args.dataset_type == 'clothing1m':
    args.version = str(args.nr) + 'nr_' + args.loss.lower()
    if args.scale_exp:
        args.version += '_scale_' + str(args.alpha)
    elif args.alpha_beta_exp:
        args.version += '_ab_' + str(args.alpha) + '_' + str(args.beta)
    if args.asym:
        log_sym_type = 'asym'
        args.version += '_asym'
        args.checkpoint_path += 'asym/' + args.run_version + '/'
    else:
        log_sym_type = 'sym'
        args.checkpoint_path += 'sym/' + args.run_version + '/'


if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)
if not os.path.exists(os.path.join('logs', log_dataset_type, log_sym_type, args.run_version)):
    os.makedirs(os.path.join('logs', log_dataset_type, log_sym_type, args.run_version))


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


log_file_name = os.path.join('logs', log_dataset_type, log_sym_type, args.run_version, args.version)
logger = setup_logger(name=args.version, log_file=log_file_name + ".log")
GLOBAL_STEP, EVAL_STEP, EVAL_BEST_ACC = 0, 0, 0
TRAIN_HISTORY = collections.defaultdict(list)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def model_eval(epoch, fixed_cnn, data_loader, dataset_type='test_dataset'):
    global EVAL_STEP
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss()

    for images, labels in tqdm(data_loader[dataset_type]):
        start = time.time()
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

        valid_loss_meters.update(loss.item(), labels.shape[0])
        valid_acc_meters.update(acc.item(), labels.shape[0])
        valid_acc5_meters.update(acc5.item(), labels.shape[0])
        end = time.time()

        EVAL_STEP += 1
        if EVAL_STEP % args.train_log_every == 0:
            display = log_display(epoch=epoch,
                                  global_step=GLOBAL_STEP,
                                  time_elapse=end-start,
                                  loss=loss.item(),
                                  test_loss_avg=valid_loss_meters.avg,
                                  acc=acc.item(),
                                  test_acc_avg=valid_acc_meters.avg,
                                  test_acc_top5_avg=valid_acc5_meters.avg)
            logger.info(display)
    display = log_display(epoch=epoch,
                          global_step=GLOBAL_STEP,
                          time_elapse=end-start,
                          loss=loss.item(),
                          test_loss_avg=valid_loss_meters.avg,
                          acc=acc.item(),
                          test_acc_avg=valid_acc_meters.avg,
                          test_acc_top5_avg=valid_acc5_meters.avg)
    logger.info(display)
    return valid_acc_meters.avg, valid_acc5_meters.avg


def train_fixed(starting_epoch, data_loader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper):
    global GLOBAL_STEP, reduction_arc, cell_arc, EVAL_BEST_ACC, EVAL_STEP, TRAIN_HISTORY

    for epoch in tqdm(range(starting_epoch, args.epoch)):
        logger.info("=" * 20 + "Training" + "=" * 20)
        fixed_cnn.train()
        train_loss_meters = AverageMeter()
        train_acc_meters = AverageMeter()
        train_acc5_meters = AverageMeter()

        for images, labels in tqdm(data_loader["train_dataset"]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            start = time.time()
            fixed_cnn.zero_grad()
            fixed_cnn_optmizer.zero_grad()
            pred = fixed_cnn(images)
            loss = criterion(pred, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.grad_bound)
            fixed_cnn_optmizer.step()
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

            train_loss_meters.update(loss.item(), labels.shape[0])
            train_acc_meters.update(acc.item(), labels.shape[0])
            train_acc5_meters.update(acc5.item(), labels.shape[0])

            end = time.time()

            GLOBAL_STEP += 1
            if GLOBAL_STEP % args.train_log_every == 0:
                lr = fixed_cnn_optmizer.param_groups[0]['lr']
                display = log_display(epoch=epoch,
                                      global_step=GLOBAL_STEP,
                                      time_elapse=end-start,
                                      loss=loss.item(),
                                      loss_avg=train_loss_meters.avg,
                                      acc=acc.item(),
                                      acc_top1_avg=train_acc_meters.avg,
                                      acc_top5_avg=train_acc5_meters.avg,
                                      lr=lr,
                                      gn=grad_norm)
                logger.info(display)
        if fixed_cnn_scheduler is not None:
            fixed_cnn_scheduler.step()
        logger.info("="*20 + "Eval" + "="*20)
        curr_acc, _ = model_eval(epoch, fixed_cnn, data_loader)
        logger.info("curr_acc\t%.4f" % curr_acc)
        logger.info("BEST_ACC\t%.4f" % EVAL_BEST_ACC)
        payload = '=' * 10 + '\n'
        payload = payload + ("curr_acc: %.4f\n best_acc: %.4f\n" % (curr_acc, EVAL_BEST_ACC))
        EVAL_BEST_ACC = max(curr_acc, EVAL_BEST_ACC)
        TRAIN_HISTORY["train_loss"].append(train_loss_meters.avg)
        TRAIN_HISTORY["train_acc"].append(train_acc_meters.avg)
        TRAIN_HISTORY["test_acc"].append(curr_acc)
        TRAIN_HISTORY["test_acc_best"] = [EVAL_BEST_ACC]
        with open(args.checkpoint_path + args.version + '.pickle', 'wb') as handle:
            pickle.dump(TRAIN_HISTORY, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved!\n")
    return


def train():
    # Dataset
    if args.dataset_type == 'clothing1m':
        dataset = Clothing1MDatasetLoader(batchSize=args.batch_size,
                                          dataPath=args.data_path,
                                          numOfWorkers=args.data_nums_workers)
    elif args.dataset_type == 'imagenet':
        dataset = ImageNetDatasetLoader(batchSize=args.batch_size,
                                        dataPath=args.data_path,
                                        seed=args.seed,
                                        target_class_num=200,
                                        nosiy_rate=0.4,
                                        numOfWorkers=args.data_nums_workers)
    else:
        dataset = DatasetGenerator(batchSize=args.batch_size,
                                   dataPath=args.data_path,
                                   numOfWorkers=args.data_nums_workers,
                                   noise_rate=args.nr,
                                   asym=args.asym,
                                   seed=args.seed,
                                   dataset_type=args.dataset_type)

    dataLoader = dataset.getDataLoader()
    eta_min = 0
    ln_neg = 1

    if args.dataset_type == 'clothing1m':
        # Train Clothing1M
        args.epoch = 20
        args.l2_reg = 1e-3
        num_classes = 14
        fixed_cnn = torchvision.models.resnet50(num_classes=14)
        # fixed_cnn.fc = torch.nn.Linear(2048, 14)

    elif args.dataset_type == 'cifar100':
        # Train CIFAR100
        args.lr = 0.1
        args.epoch = 200
        num_classes = 100
        fixed_cnn = ResNet34(num_classes=num_classes)

        # NLNL
        if args.loss == 'NLNL':
            args.epoch = 2000
            ln_neg = 110

    elif args.dataset_type == 'cifar10':
        # Train CIFAR10
        args.epoch = 120
        num_classes = 10
        fixed_cnn = SCEModel(type='cifar10')

        # NLNL
        if args.loss == 'NLNL':
            args.epoch = 1000

    elif args.dataset_type == 'mnist':
        # Train mnist
        args.epoch = 50
        num_classes = 10
        fixed_cnn = SCEModel(type='mnist')
        eta_min = 0.001
        args.l2_reg = 1e-3
        # NLNL
        if args.loss == 'NLNL':
            args.epoch = 720

    elif args.dataset_type == 'imagenet':
        args.epoch = 100
        args.l2_reg = 3e-5
        num_classes = 200
        fixed_cnn = torchvision.models.resnet50(num_classes=num_classes)

    logger.info("num_classes: %s" % (num_classes))

    loss_options = {
        'SCE': SCELoss(alpha=args.alpha, beta=args.beta, num_classes=num_classes),
        'CE': torch.nn.CrossEntropyLoss(),
        'NCE': NormalizedCrossEntropy(scale=args.alpha, num_classes=num_classes),
        'MAE': MeanAbsoluteError(scale=args.alpha, num_classes=num_classes),
        'NMAE': NormalizedMeanAbsoluteError(scale=args.alpha, num_classes=num_classes),
        'GCE': GeneralizedCrossEntropy(num_classes=num_classes, q=args.q),
        'RCE': ReverseCrossEntropy(scale=args.alpha, num_classes=num_classes),
        'NRCE': NormalizedReverseCrossEntropy(scale=args.alpha, num_classes=num_classes),
        'NGCE': NormalizedGeneralizedCrossEntropy(scale=args.alpha, num_classes=num_classes, q=args.q),
        'NCEandRCE': NCEandRCE(alpha=args.alpha, beta=args.beta, num_classes=num_classes),
        'NCEandMAE': NCEandMAE(alpha=args.alpha, beta=args.beta, num_classes=num_classes),
        'GCEandMAE': GCEandMAE(alpha=args.alpha, beta=args.beta, num_classes=num_classes, q=args.q),
        'GCEandRCE': GCEandRCE(alpha=args.alpha, beta=args.beta, num_classes=num_classes, q=args.q),
        'GCEandNCE': GCEandNCE(alpha=args.alpha, beta=args.beta, num_classes=num_classes, q=args.q),
        'MAEandRCE': MAEandRCE(alpha=args.alpha, beta=args.beta, num_classes=num_classes),
        'NGCEandNCE': NGCEandNCE(alpha=args.alpha, beta=args.beta, num_classes=num_classes, q=args.q),
        'NGCEandMAE': NGCEandMAE(alpha=args.alpha, beta=args.beta, num_classes=num_classes, q=args.q),
        'NGCEandRCE': NGCEandRCE(alpha=args.alpha, beta=args.beta, num_classes=num_classes, q=args.q),
        'FocalLoss': FocalLoss(gamma=args.gamma),
        'NFL': NormalizedFocalLoss(scale=args.alpha, gamma=args.gamma, num_classes=num_classes),
        'NLNL': NLNL(num_classes=num_classes, train_loader=dataLoader['train_dataset'], ln_neg=ln_neg),
        'NFLandNCE': NFLandNCE(alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_classes=num_classes),
        'NFLandMAE': NFLandMAE(alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_classes=num_classes),
        'NFLandRCE': NFLandRCE(alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_classes=num_classes),
        'DMI': DMILoss(num_classes=num_classes)
    }

    if args.loss in loss_options:
        criterion = loss_options[args.loss]
    else:
        raise("Unknown loss")

    logger.info(criterion.__class__.__name__)
    logger.info("Number of Trainable Parameters %.4f" % count_parameters_in_MB(fixed_cnn))

    fixed_cnn.to(device)

    if args.loss == 'DMI':
        criterion = loss_options['CE']

    fixed_cnn_optmizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         weight_decay=args.l2_reg)

    fixed_cnn_scheduler = CosineAnnealingLR(fixed_cnn_optmizer,
                                            float(args.epoch),
                                            eta_min=eta_min)
    if args.dataset_type == 'clothing1m':
        fixed_cnn_scheduler = MultiStepLR(fixed_cnn_optmizer, milestones=[5, 10], gamma=0.1)
    elif args.dataset_type == 'imagenet':
        fixed_cnn_scheduler = MultiStepLR(fixed_cnn_optmizer, milestones=[30, 60, 80], gamma=0.1)

    utilHelper = TrainUtil(checkpoint_path=args.checkpoint_path, version=args.version)
    starting_epoch = 0

    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))

    train_fixed(starting_epoch, dataLoader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper)

    if args.loss == 'DMI':
        criterion = loss_options['DMI']
        fixed_cnn_optmizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                             lr=1e-6,
                                             momentum=0.9,
                                             weight_decay=args.l2_reg)
        starting_epoch = 0
        fixed_cnn_scheduler = None
        train_fixed(starting_epoch, dataLoader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper)


if __name__ == '__main__':
    train()
