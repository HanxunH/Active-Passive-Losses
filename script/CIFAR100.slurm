#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:1

# The project ID which this job should run under:
#SBATCH --account="punim0784"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# The amount of memory in megabytes per process in the job:
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time 168:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Run the job from this directory:
cd /data/cephfs/punim0784/robust_loss_nips


# The modules to load:
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA10
nvidia-smi

exp_name=$1
seed=$2
loss=$3


# Sym
declare -a nr_arr=("0.0"
                   "0.2"
                   "0.4"
                   "0.6"
                   "0.8")

for i in "${nr_arr[@]}"
    do
    rm -rf ${exp_name}/cifar100/sym/$i/${loss}/*
    python3 -u main.py --exp_name ${exp_name}/cifar100/sym/$i --seed $seed --noise_rate $i --config_path configs/cifar100/sym --version ${loss}
done

# Asym
declare -a nr_arr=(
                   "0.1"
                   "0.2"
                   "0.3"
                   "0.4"
                  )
for i in "${nr_arr[@]}"
    do
      rm -rf ${exp_name}/cifar100/asym/$i/${loss}/*
      python3 -u main.py --exp_name ${exp_name}/cifar100/asym/$i --seed $seed --noise_rate $i --config_path configs/cifar100/asym --version ${loss} --asym
done
