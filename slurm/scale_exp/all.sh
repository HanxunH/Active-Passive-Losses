#!/bin/bash


declare -a loss_arr=("nfl")

#
declare -a alpha_arr=(1.0
                      2.0
                      4.0
                      5.0
                      10.0)

for i in "${loss_arr[@]}"
do
  for k in "${alpha_arr[@]}"
  do
    job_name=${i}${j}-C100
    echo ${i}.slurm $1 $2 $k
    sbatch --partition deeplearn --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2 $k
  done
done
