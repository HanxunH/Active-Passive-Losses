#!/bin/bash

# Run on Shortgpgpu
# declare -a loss_arr=("ce"
#                      "gce"
#                      "gce_mae"
#                      "gce_nce"
#                      "gce_rce"
#                      "mae"
#                      "nce"
#                      "nce_mae"
#                      "nce_rce"
#                      "ngce"
#                      "ngce_mae"
#                      "ngce_rce"
#                      "ngce_nce"
#                      "rce"
#                      "sce")

declare -a loss_arr=("nrce")

declare -a nr_arr=("0.0"
                   "0.2"
                   "0.4"
                   "0.6"
                   "0.8")

for i in "${loss_arr[@]}"
do
  for j in "${nr_arr[@]}"
  do
    job_name=${i}${j}-C10
    sbatch --partition deeplearn --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $j $1 $2
    # sbatch --partition shortgpgpu --time 1:00:00 --cpus-per-task=1 --job-name $job_name --mem=16G --gres=gpu:p100:1 ${i}.slurm $j $1 $2
    # or do whatever with individual element of the array
  done
done

# Run on gpgpgpu
# declare -a loss_arr=("nlnl")
#
# for i in "${loss_arr[@]}"
# do
#   for j in "${nr_arr[@]}"
#   do
#     job_name=${i}${j}-C10
#     sbatch --partition deeplearn --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $j $1 $2
#     # or do whatever with individual element of the array
#   done
# done
