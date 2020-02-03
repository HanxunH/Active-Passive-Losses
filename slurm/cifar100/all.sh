#!/bin/bash


# declare -a loss_arr=("dmi"
#                     )
#
# for i in "${loss_arr[@]}"
# do
#   job_name=${i}_${1}
#   echo ${i}.slurm $1 $2
#   sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2
# done


declare -a loss_arr=("nlnl")


declare -a sym_nr_arr=("0.0"
                       "0.2"
                       "0.4"
                       "0.6"
                       "0.8")

declare -a asym_nr_arr=("0.1"
                        "0.2"
                        "0.3"
                        "0.4")


# for i in "${loss_arr[@]}"
# do
#   echo $i
#   for j in "${asym_nr_arr[@]}"
#   do
#     job_name=${i}_${1}
#     echo ${i}.slurm $1 $2 $j
#     sbatch --partition gpgpu --cpus-per-task=2 --gres=gpu:p100:1 --job-name $job_name --mem=16G ${i}.slurm $1 $2 $j
#     # sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2 $j
#   done
# done

for j in "${asym_nr_arr[@]}"
  do
    job_name=nlnl_asym_${1}
    echo nlnl_asym.slurm $1 $2 $j
    sbatch --partition gpgpu --cpus-per-task=2 --gres=gpu:p100:1 --job-name $job_name --mem=16G nlnl_asym.slurm $1 $2 $j
    # sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2 $j
done

for j in "${sym_nr_arr[@]}"
  do
    job_name=nlnl_${1}
    echo nlnl.slurm $1 $2 $j
    sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G nlnl.slurm $1 $2 $j
done
