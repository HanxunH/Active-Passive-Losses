import torch
import argparse
import util
import os
import datetime
import random
import mlconfig
import loss
import models
import dataset
import shutil
from evaluator import Evaluator
from trainer import Trainer

# ArgParse
parser = argparse.ArgumentParser(description='Normalized Loss Functions for Deep Learning with Noisy Labels')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config_path', type=str, default='configs')
parser.add_argument('--version', type=str, default='ce')
parser.add_argument('--exp_name', type=str, default="run1")
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--noise_rate', type=float, default=0.0)
args = parser.parse_args()

# Set up
if args.exp_name == '' or args.exp_name is None:
    args.exp_name = 'exp_' + datetime.datetime.now()
exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)

logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    logger.info("Using CUDA!")
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

logger.info("PyTorch Version: %s" % (torch.__version__))
config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
for key in config:
    logger.info("%s: %s" % (key, config[key]))


def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("="*20 + "Training" + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, ENV['global_step'], model, optimizer, criterion)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval" + "="*20)
        evaluator.eval(epoch, ENV['global_step'], model, torch.nn.CrossEntropyLoss())
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100
        ENV['best_acc'] = max(ENV['curren_acc'], ENV['best_acc'])

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
    return


def main():
    if config.dataset.name == 'DatasetGenerator':
        data_loader = config.dataset(seed=args.seed, noise_rate=args.noise_rate, asym=args.asym)
    else:
        data_loader = config.dataset()

    model = config.model()
    if isinstance(data_loader, dataset.Clothing1MDatasetLoader):
        model.fc = torch.nn.Linear(2048, 14)
    model = model.to(device)

    data_loader = data_loader.getDataLoader()
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    if config.criterion.name == 'NLNL':
        criterion = config.criterion(train_loader=data_loader['train_dataset'])
    else:
        criterion = config.criterion()
    trainer = Trainer(data_loader['train_dataset'], logger, config)
    evaluator = Evaluator(data_loader['test_dataset'], logger, config)

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'current_acc': 0.0,
           'train_history': [],
           'eval_history': []}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler=scheduler)
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV)
    return


if __name__ == '__main__':
    main()
