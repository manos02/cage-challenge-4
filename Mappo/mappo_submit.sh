#!/bin/bash
#SBATCH --job-name=mappo_ray
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=38G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=R-%x.%j.out

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/projects/cage-challenge-4/myenv/bin/activate

export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# determine GPU count (defaults to 0 if unset)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}

echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

ray start --head \
    --port=6379 \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=$NUM_GPUS

sleep 10

cd $HOME/projects/cage-challenge-4

python Mappo/mappo_hyperparameter.py --cluster


