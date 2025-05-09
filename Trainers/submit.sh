#!/bin/bash
#SBATCH --job-name=ippo_ray
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH --gpus-per-node=a100:4         
#SBATCH --time=04:00:00

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/projects/cage-challenge-4/myenv/bin/activate

ray start --head \
    --node-ip-address=$HOSTNAME \
    --port=6379 \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=$SLURM_GPUS_PER_NODE

sleep 10

python -m Trainers.ippo_hyperparameter
