# Normalized Loss Functions - Active Passive Losses
Code for ICML2020 Paper ["Normalized Loss Functions for Deep Learning with Noisy Labels"]() (Preprint Version coming soon)

## Requirements
```console
Python >= 3.6, PyTorch >= 1.3.1, torchvision >= 0.4.1, mlconfig
```

## How To Run
##### Configs for the experiment settings
Check '*.yaml' file in the config folder for each experiment.

##### Arguments
* noise_rate: noise rate
* asym: use if it is asymmetric noise, default is symmetric
* config_path: path to the configs folder
* version: the config file name
* exp_name: name of the experiments (as note)
* seed: random seed

Example for 0.4 Symmetric noise rate with NCE+RCE loss
```console
# CIFAR-10
$  python3  main.py --exp_name      test_exp            \
                    --noise_rate    0.4                 \
                    --version       nce+rce             \
                    ----config_path configs/cifar10/sym \
                    --seed          123

# CIFAR-100
$  python3  main.py --exp_name      test_exp             \
                    --noise_rate    0.4                  \
                    --version       nce+rce              \
                    ----config_path configs/cifar100/sym \
                    --seed          123
```
