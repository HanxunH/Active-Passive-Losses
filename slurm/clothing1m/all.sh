# NLNL
sbatch --partition deeplearn --gres=gpu:v100:1 --time 30:00:00 --qos gpgpudeeplearn ce.slurm
sbatch --partition deeplearn --gres=gpu:v100:1 --time 30:00:00 --qos gpgpudeeplearn sce.slurm
sbatch --partition deeplearn --gres=gpu:v100:1 --time 30:00:00 --qos gpgpudeeplearn gce.slurm
sbatch --partition deeplearn --gres=gpu:v100:1 --time 30:00:00 --qos gpgpudeeplearn nce_rce.slurm
sbatch --partition deeplearn --gres=gpu:v100:1 --time 30:00:00 --qos gpgpudeeplearn nfl_rce.slurm
