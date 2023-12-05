#!/bin/bash
#SBATCH --job-name=cub-bg-reliance # Specify a name for your job
#SBATCH --output=slurm-logs/out-cub-bg-reliance.log       # Specify the output log file
#SBATCH --error=slurm-logs/err-cub-bg-reliance.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa4000:1              # Number of GPUs per node
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

datasets=("bg-only-black" "bg-only-inpaint" "no-fg")

cd /fs/nexus-scratch/rezashkv/research/projects/BG-Analysis/CUB || exit

for dataset in "${datasets[@]}"; do
  python3 train.py --dataset "$dataset" \
                   --epochs 100 \
                   --exp_name "bg-reliance" \
                   --model resnet18 \
                   --lr 0.001 \
                   --batch_size 128 \
                   --weight_decay 1000 \
                   --save_dir "/fs/nexus-scratch/rezashkv/research/results/BG-Analysis/cub-models/" \
                   --device "cuda"
done