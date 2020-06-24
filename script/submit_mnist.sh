# MNIST
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
    job_name=${j}_MNIST_${i}
    echo $job_name
    # sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --job-name $job_name --cpus-per-task=8 --gres=gpu:1 MNIST.slurm $i $seed $j
    sbatch --partition gpgpu --job-name $job_name --cpus-per-task=4 --gres=gpu:1 --mem=16G MNIST.slurm $i $seed $j
  done
  seed=$((seed+1))
done
