#!/bin/bash
#SBATCH --job-name=in9l-vit # Specify a name for your job
#SBATCH --output=/fs/cbcb-scratch/aliganj/results/logs/out-in9-bg-reliance.log       # Specify the output log file
#SBATCH --error=/fs/cbcb-scratch/aliganj/results/logs/err-in9-bg-reliance.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=54:00:00           # Maximum execution time (HH:MM:SS)
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1              # Number of GPUs per node
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

conda_env_path="/fs/cbcb-scratch/aliganj/software/miniconda/envs/pytorch2.1.1"

source activate "$conda_env_path"

cd /fs/cbcb-scratch/aliganj/projects/bg_analysis/imagenet || exit

python3 bg_reliance.py
