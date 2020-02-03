#!/bin/bash

declare -a loss_arr=(
                      "nce_rce"
                    )

#
declare -a alpha_arr=("0.1"
                      "1.0"
                      "10.0")

declare -a beta_arr=("0.1"
                     "1.0"
                     "10.0"
                     "100.0")

for i in "${loss_arr[@]}"
do
  for alpha in "${alpha_arr[@]}"
  do
    for beta in "${beta_arr[@]}"
    do
      job_name=${i}_${1}
      echo ${i}.slurm $1 $2 $alpha $beta
      sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2 $alpha $beta
    done
  done
done
