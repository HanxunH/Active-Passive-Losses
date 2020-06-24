# Normalized Loss Functions - Active Passive Losses
Code for ICML2020 Paper ["Normalized Loss Functions for Deep Learning with Noisy Labels"]()(Preprint Version comming soon)

## Requirements
```console
Python >= 3.6, PyTorch >= 1.3.1, torchvision >= 0.4.1
```

## How To Run
##### Arguments

* loss: Options for the loss function. Complete list can be found in train.py
* nr: Noise rates
* asym: Run with asymmetric noise
* dataset_type: Options for the dataset ["mnist", "cifar10", "cifar100"]
* data_path: Path to the dataset dir
* alpha: Alpha parameter for the loss function
* beta: Beta parameter for the loss function
* l2_reg: L2 weight decays

Example for 0.4 Symmetric noise rate with NCE+RCE loss
```console
# CIFAR-10
$  python3  train.py   --nr      0.4         \
                       --loss    NCEandRCE   \
                       --run_version run1    \
                       --seed        123

# CIFAR-100
$  python3  train.py   --nr      0.4         \
                       --loss    NCEandRCE   \
                       --alpha   10.0        \
                       --beta    0.1         \
                       --l2_reg              1e-5        \
                       --dataset_type        cifar100    \
                       --run_version         run1        \
                       --seed                123

```
