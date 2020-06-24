# WebVision Mini
declare -a loss=(
                  "ce"
                  "gce"
                  # "nce+mae"
                  # "nce+rce"
                  # "nfl+mae"
                  # "nfl+rce"
                  "sce"
                  )

declare -a run_version=(
                        "webvision_mini"
                        )

seed=0
for i in "${run_version[@]}"
do
  for j in "${loss[@]}"
  do
    job_name=WebVisionMini${i}_${j}
    echo $job_name
    sbatch --partition gpgpu --job-name $job_name --cpus-per-task=8 --gres=gpu:4 --mem=96G WebVisionMini.slurm $i $seed $j
    # sbatch --partition deeplearn --qos gpgpudeeplearn --job-name $job_name --cpus-per-task=8 --gres=gpu:4 --mem=96G WebVisionMini.slurm $i $seed $j
    # sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --job-name $job_name --cpus-per-task=24 --gres=gpu:4 WebVisionMini.slurm $i $seed $j
  done
  seed=$((seed+1))
done
