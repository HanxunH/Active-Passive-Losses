import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
sns.set_style("whitegrid")

sym_nr_rates = ['0.0', '0.2', '0.4', '0.6', '0.8']

loss_style = {
    'CE': {'color': 'b', 'linestyle': '-'},
    'GCE': {'color': 'r', 'linestyle': '-'},
    'RCE': {'color': 'c', 'linestyle': '-'},
    'NRCE': {'color': 'c', 'linestyle': '--'},
    'GCE+MAE': {'color': 'palegreen', 'linestyle': '-'},
    'GCE+RCE': {'color': 'orange', 'linestyle': '-'},
    'MAE': {'color': 'm', 'linestyle': '-'},
    'NMAE': {'color': 'm', 'linestyle': '--'},
    'NCE': {'color': 'b', 'linestyle': '--'},
    'NCE+MAE': {'color': 'plum', 'linestyle': '--'},
    'NCE+RCE': {'color': 'slateblue', 'linestyle': '--'},
    'NGCE': {'color': 'r', 'linestyle': '--'},
    'NGCE+RCE': {'color': 'salmon', 'linestyle': '--'},
    'NGCE+MAE': {'color': 'deeppink', 'linestyle': '--'},
    'NGCE+NCE': {'color': 'limegreen', 'linestyle': '--'},
}


def load_train_history(target_dir_list):
    history = defaultdict(lambda: defaultdict(list))
    for target_dir in target_dir_list:
        for file in sorted(os.listdir(target_dir)):
            file_path = os.path.join(target_dir, file)
            file = os.path.splitext(file)[0]
            file = file.replace('nr', '')
            file = file.split('_')
            if 'scale' in file:
                nr, loss_name, _, scale_rate = file[0], file[1], file[2], file[3]
                loss_name = loss_name + ' Scale=' + scale_rate
            else:
                nr, loss_name = file[0], file[1]
            loss_name = loss_name.replace('and', '+')
            loss_name = loss_name.upper()
            if loss_name == 'NLNL':
                continue
            with open(file_path, 'rb') as handle:
                train_history = pickle.load(handle)
            history[loss_name][nr].append(train_history)

    # Compute Avg and std
    for loss_name in history:
        for nr in history[loss_name]:
            test_acc_arr = []
            for arr in history[loss_name][nr]:
                test_acc_arr.append(arr['test_acc'])

            prev = None
            for i, item in enumerate(test_acc_arr):
                if prev is None:
                    prev = len(item)
                elif prev != len(item):
                    print(loss_name, nr, i)

            test_acc_arr = np.asarray(test_acc_arr)
            history[loss_name][nr].append(test_acc_arr.mean(axis=0))
            history[loss_name][nr].append(test_acc_arr.std(axis=0))
    return history


def plot_graph_pair_compare(history, nr, targets, graph_file_name, range_min, range_max):
    '''
        graph1 cifar10-0.8 no + comb, just CE GCE RCE MAE +/ NCE NGCE NRCE NMAE + color red blue cayan(red purple) b c r m
               split by 4 1v1 remove title
    '''
    for loss, normal_loss in targets:
        graph_name = loss + '_vs_' + normal_loss
        fig = plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        ax.set_xlabel("Epochs", fontsize=25)
        ax.set_ylabel('Test Accuracy', fontsize=25)
        ax.set_ylim(bottom=range_min, top=range_max)

        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.grid(True, axis='x')
        ax.grid(True, axis='y')

        ax.plot(history[loss][nr][-2],
                color=loss_style[loss]['color'],
                linestyle=loss_style[loss]['linestyle'],
                linewidth=4,
                alpha=0.8,
                label=loss)

        ax.plot(history[normal_loss][nr][-2],
                color=loss_style[normal_loss]['color'],
                linestyle=loss_style[normal_loss]['linestyle'],
                linewidth=4,
                alpha=0.8,
                label=normal_loss)
        ax.legend(ncol=1, fontsize=22, loc='lower right')
        plt.savefig(graph_file_name + '_' + graph_name + '.png', bbox_inches='tight')
    return


def plot_graph_compare(history, nr, targets, graph_file_name, range_min, range_max):
    fig = plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epochs", fontsize=25)
    ax.set_ylabel('Test Accuracy', fontsize=25)
    ax.set_ylim(bottom=range_min, top=range_max)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.grid(True, axis='x')
    ax.grid(True, axis='y')

    for loss_name in targets:
        ax.plot(history[loss_name][nr][-2],
                color=loss_style[loss_name]['color'],
                linestyle=loss_style[loss_name]['linestyle'],
                linewidth=4,
                alpha=0.8,
                label=loss_name)
    ax.legend(ncol=1, fontsize=22, loc='lower right')
    plt.savefig(graph_file_name + '.png', bbox_inches='tight')
    return


# CIFAR-10 Sym
target_dir_list = ['checkpoints/cifar10/sym/run1']
cifar_10_history_sym = load_train_history(target_dir_list)

# CIFAR-10 Asym
target_dir_list = ['checkpoints/cifar10/asym/run1']
cifar_10_history_asym = load_train_history(target_dir_list)


# CIFAR-100 Sym
target_dir_list = ['checkpoints/cifar100/sym/run1']
cifar_100_history_sym = load_train_history(target_dir_list)


# CIFAR-100 Asym
target_dir_list = ['checkpoints/cifar100/asym/run1']
cifar_100_history_asym = load_train_history(target_dir_list)


# '''
#     Plot Graph 1
# '''
# plot_graph_pair_compare(history=cifar_10_history_sym,
#                         nr='0.6',
#                         range_min=0.3,
#                         range_max=0.9,
#                         graph_file_name='plot/graph1',
#                         targets=[('CE', 'NCE'), ('GCE', 'NGCE'), ('MAE', 'NMAE'), ('RCE', 'NRCE')])
#
# '''
#     Plot Graph 2
# '''
# plot_graph_pair_compare(history=cifar_100_history_sym,
#                         nr='0.6',
#                         range_min=0.0,
#                         range_max=0.5,
#                         graph_file_name='plot/graph2',
#                         targets=[('CE', 'NCE'), ('GCE', 'NGCE'), ('MAE', 'NMAE'), ('RCE', 'NRCE')])


'''
    Plot Graph 3
'''
target_dir_list = ['checkpoints/cifar100/sym/scale_exp']
cifar_100_history_scale_exp = load_train_history(target_dir_list)
for item in sym_nr_rates:
    cifar_100_history_scale_exp['CE'][item] = cifar_100_history_sym['CE'][item]

plot_graph_compare(history=cifar_100_history_scale_exp,
                   nr='0.6',
                   range_min=0.0,
                   range_max=0.5,
                   graph_file_name='plot/graph3_scale_NCE',
                   targets=['CE', 'NCE Scale=1.0', 'NCE Scale=5.0', 'NCE Scale=10.0'])


'''
    Plot Graph 4
'''
plot_graph_compare(history=cifar_100_history_sym,
                   nr='0.6',
                   range_min=0.0,
                   range_max=0.5,
                   graph_file_name='plot/graph4_cifar_100_0.6',
                   targets=['NCE+MAE', 'NCE+RCE', 'NGCE+RCE', 'NGCE+MAE', 'NGCE+NCE'])


plot_graph_compare(history=cifar_10_history_sym,
                   nr='0.6',
                   range_min=0.0,
                   range_max=0.9,
                   graph_file_name='plot/graph4_cifar_10_0.6',
                   targets=['NCE+RCE', 'NCE+MAE', 'NGCE+RCE', 'NGCE+MAE', 'NGCE+NCE'])

'''
    Notes
    y = 'test accuracy'
    graph1 cifar10-0.8 no + comb, just CE GCE RCE MAE +/ NCE NGCE NRCE NMAE + color red blue cayan(red purple) b c r m
           split by 4 1v1 remove title
    grpah2 cifar100-0.6 graph1... no graph2
    graph3 scale NCE NGCE RCE MAE Scale by 1 2 4 (5 6 8 10) select 3? split by loss add 'CE as black --' or 'GCE'
    graph4 +comb cifar-10 0.8 cifar-100 0.6 good(-) not good(--) order legend by good -> not good
    graph5 cifar10/cifar100 test cifar alpha beta scale by lmbda
    gce rce remove only normalized
'''
