#!/bin/bash
#SBATCH --job-name=ippo_ray
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00   

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/projects/cage-challenge-4/myenv/bin/activate

export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# determine GPU count (defaults to 0 if unset)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}


echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi


ray start --head \
    --node-ip-address=$HOSTNAME \
    --port=6379 \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=$NUM_GPUS

sleep 10

cd $HOME/projects/cage-challenge-4

python -m Ippo.ippo_hyperparameter