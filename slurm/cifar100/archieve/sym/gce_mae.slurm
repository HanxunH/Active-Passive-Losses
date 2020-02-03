#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Sat Mar 02 2019 00:19:42 GMT+1100 (Australian Eastern Daylight Time)
# Multithreaded (SMP) job: must run on one node and the cloud partition
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1

# The project ID which this job should run under:
#SBATCH --account="punim0784"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# The amount of memory in megabytes per process in the job:
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time 4:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Run the job from this directory:
cd /data/cephfs/punim0784/robust_loss

# The modules to load:
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA10
nvidia-smi

mkdir -p logs/cifar100/sym/${2}

python3 -u train.py   --nr      $1          \
                      --l2_reg  1e-5        \
                      --loss    GCEandMAE   \
                      --dataset_type  cifar100      \
                      --run_version $2      \
                      --seed    $3
