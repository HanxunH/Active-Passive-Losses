# CIFAR10
declare -a loss=(
                  "ce"
                  "focal"
                  "gce"
                  "mae"
                  "nce"
                  "nce+mae"
                  "nce+rce"
                  "nfl"
                  "nfl+mae"
                  "nfl+rce"
                  "ngce"
                  "ngce+mae"
                  "ngce+rce"
                  # "nlnl"
                  "rce"
                  "sce"
                )

declare -a run_version=(
                        "run1"
                        "run2"
                        "run3"
                       )

seed=0
for i in "${run_version[@]}"
do
  for j in "${loss[@]}"
  do
    echo C100_${i}_${j}
    job_name=${j}_C100_${i}
    # sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --job-name $job_name --cpus-per-task=8 --gres=gpu:1 CIFAR10.slurm $i $seed $j
    sbatch --partition gpgpu --job-name $job_name --cpus-per-task=4 --gres=gpu:1 --mem=32G CIFAR100.slurm $i $seed $j
    # sbatch --partition deeplearn --qos gpgpudeeplearn --job-name $job_name --cpus-per-task=4 --gres=gpu:1 --mem=32G CIFAR10.slurm $i $seed $j
  done
  seed=$((seed+1))
done
