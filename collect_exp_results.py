import os
import torch
import collections
import json
import statistics
import util
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint_history(checkpoint_path):
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pth"):
            checkpoint = torch.load(os.path.join(checkpoint_path, file), map_location=device)
            return checkpoint['ENV']


def load_dataset_results(exp_results, dataset_target, exp_name, noise_rate_type=None):
    target_path = os.path.join(dataset_target, noise_rate_type)
    for noise_rate in os.listdir(target_path):
        path_with_noise_rate = os.path.join(target_path, noise_rate)
        for loss_name in os.listdir(path_with_noise_rate):
            checkpoint_path = os.path.join(path_with_noise_rate, loss_name, 'checkpoints')
            try:
                ENV = load_checkpoint_history(checkpoint_path)
                process_results(exp_results, loss_name, exp_name, noise_rate, ENV)
            except Exception as e:
                print(loss_name, noise_rate, exp_name, str(e))
    return


def process_results(exp_results, loss_name, exp_name, noise_rate, ENV):
    if loss_name not in exp_results:
        exp_results[loss_name] = {}
    if noise_rate not in exp_results[loss_name]:
        exp_results[loss_name][noise_rate] = {}
    if exp_name not in exp_results[loss_name][noise_rate]:
        exp_results[loss_name][noise_rate][exp_name] = {'ENV': ENV, 'last_acc': ENV['eval_history'][-1]}
    return


def process_avg_table(exp_results):
    for loss_name in exp_results:
        for noise_rate in exp_results[loss_name]:
            running_sum = []
            for exp_name in exp_results[loss_name][noise_rate]:
                running_sum.append(exp_results[loss_name][noise_rate][exp_name]['last_acc'])
            if len(running_sum) == 0:
                continue
            mean_acc = sum(running_sum)/len(running_sum)
            std = statistics.stdev(running_sum) if len(running_sum) > 1 else 0
            exp_results[loss_name][noise_rate]['avg_last_acc'] = mean_acc
            exp_results[loss_name][noise_rate]['std_last_acc'] = std


def load_results(dataset_name, noise_rate_type, exp_names):
    exp_results = collections.defaultdict(dict)
    exp_results_file_name = '%s_%s_exp_results.json' % (dataset_name, noise_rate_type)
    for exp in exp_names:
        path = os.path.join(exp)
        target_data_set_exp = os.path.join(path, dataset_name)
        load_dataset_results(exp_results=exp_results,
                             dataset_target=target_data_set_exp,
                             exp_name=exp,
                             noise_rate_type=noise_rate_type)
    process_avg_table(exp_results)
    with open(os.path.join('results', exp_results_file_name), 'w') as outfile:
        json.dump(exp_results, outfile)


if __name__ == '__main__':
    util.build_dirs('results')
    dataset_names = ['mnist', 'cifar10', 'cifar100']
    noise_rate_types = ['sym', 'asym']
    exp_names = ['run1', 'run2', 'run3']

    for dataset_name in dataset_names:
        for noise_rate_type in noise_rate_types:
            load_results(dataset_name, noise_rate_type, exp_names)
