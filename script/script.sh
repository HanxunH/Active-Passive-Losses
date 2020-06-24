#
# # CIFAR100
# declare -a loss=( "ce"
#                   "focal"
#                   "gce"
#                   "mae"
#                   "nce"
#                   "nce+mae"
#                   "nce+rce"
#                   "nfl"
#                   "nfl+mae"
#                   "nfl+rce"
#                   "ngce"
#                   "ngce+mae"
#                   "ngce+rce"
#                   "nlnl"
#                   "rce"
#                   "sce" )
#
# declare -a run_version=(
#                         "run1"
#                         "run2"
#                         "run3"
#                        )
#
# seed=0
# for i in "${run_version[@]}"
# do
#   for j in "${loss[@]}"
#   do
#     job_name=C100_${i}_${j}
#     echo $job_name
#     sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --job-name $job_name --cpus-per-task=8 --gres=gpu:1 CIFAR100.slurm $i $seed $j
#   done
#   seed=$((seed+1))
# done


# # WebVision Full
# declare -a loss=(
#                   "ce"
#                   "gce"
#                   "nce+mae"
#                   "nce+rce"
#                   "nfl+mae"
#                   "nfl+rce"
#                   "sce"
#                 )
#
# declare -a run_version=(
#                          "webvision_full"
#                        )
#
# seed=0
# for i in "${run_version[@]}"
# do
#   for j in "${loss[@]}"
#   do
#     job_name=WebVisionFull_${i}_${j}
#     echo $job_name
#     sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --job-name $job_name --cpus-per-task=12 --gres=gpu:4 WebVisionFull.slurm $i $seed $j
#   done
#   seed=$((seed+1))
# done

# # WebVision Mini
# declare -a loss=(
#                   "ce"
#                   "gce"
#                   "nce+mae"
#                   "nce+rce"
#                   "nfl+mae"
#                   "nfl+rce"
#                   "sce"
#                   )
#
# declare -a run_version=(
#                         "webvision_mini"
#                         )
#
# seed=0
# for i in "${run_version[@]}"
# do
#   for j in "${loss[@]}"
#   do
#     job_name=WebVisionMini${i}_${j}
#     echo $job_name
#     sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --job-name $job_name --cpus-per-task=24 --gres=gpu:4 WebVisionMini.slurm $i $seed $j
#   done
#   seed=$((seed+1))
# done
