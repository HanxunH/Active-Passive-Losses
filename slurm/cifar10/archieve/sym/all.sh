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

declare -a loss_arr=("nfocal")

for i in "${loss_arr[@]}"
do
  job_name=${i}_${1}
  sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2
done
