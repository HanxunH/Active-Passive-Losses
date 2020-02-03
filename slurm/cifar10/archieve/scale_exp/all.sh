#!/bin/bash


declare -a loss_arr=("ngce"
                     "nce"
                     "mae"
                     "rce")

#
declare -a alpha_arr=(1.0
                      2.0
                      4.0
                      5.0
                      10.0)

declare -a nr_arr=("0.0"
                   "0.2"
                   "0.4"
                   "0.6"
                   "0.8")

for i in "${loss_arr[@]}"
do
  for j in "${nr_arr[@]}"
  do
    for k in "${alpha_arr[@]}"
    do
      job_name=${i}${j}-C100
      sbatch --partition deeplearn --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $j $1 $2 $k
      # sbatch --partition gpgpu --gres=gpu:p100:1 --job-name $job_name --mem=16G ${i}.slurm $j $1 $2
    done
  done
done
