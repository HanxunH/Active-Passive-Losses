#!/bin/bash


declare -a loss_arr=(
                      "nfocal"
                    )


declare -a nr_arr=(
                   "0.1"
                   "0.2"
                   "0.3"
                  )

for i in "${loss_arr[@]}"
do
  for j in "${nr_arr[@]}"
  do
    job_name=${i}${j}-C100
    sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $j $1 $2
    # sbatch --partition gpgpu --gres=gpu:p100:1 --job-name $job_name --mem=16G ${i}.slurm $j $1 $2
  done
done
