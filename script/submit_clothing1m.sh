# Clothing1M
declare -a loss=( "ce"
                  "gce"
                  "nce+mae"
                  "nce+rce"
                  "nfl+mae"
                  "nfl+rce"
                  "sce" )

declare -a run_version=(
                        "clothing1m"
                       )

seed=0
for i in "${run_version[@]}"
do
  for j in "${loss[@]}"
  do
    job_name=Clothing1M_${i}_${j}
    echo $job_name
    sbatch --partition gpgpu --cpus-per-task=8 --gres=gpu:4 Clothing1M.slurm $i $seed $j
    # sbatch --partition gpgputest --qos=gpgpuhpcadmingpgpu --cpus-per-task=24 --gres=gpu:4 Clothing1M.slurm $i $seed $j
  done
  seed=$((seed+1))
done
