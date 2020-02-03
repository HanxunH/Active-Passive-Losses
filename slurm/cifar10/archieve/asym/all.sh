declare -a loss_arr=("nfocal")

for i in "${loss_arr[@]}"
do
  job_name=${i}_${1}
  sbatch --partition deeplearn --cpus-per-task=2 --gres=gpu:v100:1 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${i}.slurm $1 $2
done
