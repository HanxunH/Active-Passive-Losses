# Active Passive Losses
Code for Paper ["Rethinking Robust Loss Functions for Deep Learning with Noisy Labels"]()

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

Example for 0.4 Symmetric noise rate with NCE+RCE loss on CIFAR-100
```console
$  python3  train.py   --nr      0.4         \
                       --loss    NCEandRCE   \
                       --alpha   10.0        \
                       --beta    0.1         \
                       --l2_reg              1e-5        \
                       --dataset_type        cifar100    \
                       --run_version         run1        \
                       --seed                123
```
