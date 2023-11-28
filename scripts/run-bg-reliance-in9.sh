#!/bin/bash
#SBATCH --job-name=in9-bg-reliance # Specify a name for your job
#SBATCH --output=slurm-logs/out-in9-bg-reliance.log       # Specify the output log file
#SBATCH --error=slurm-logs/err-in9-bg-reliance.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00           # Maximum execution time (HH:MM:SS)
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1              # Number of GPUs per node
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

dataset_variations=("only_bg_b" "only_bg_t" "no_fg")
exp_name="bg-reliance"
dataset="in9"

cd /path/to/bg_analysis/Imagenet || exit

for dataset_v in "${dataset_variations[@]}"; do
  python3 bg_reliance.py --dataset_dir '/path/to/in9' \
    --dataset "$dataset_v" \
    --epochs 100 \
    --exp_name "${exp_name}-${dataset}" \
    --model vit \
    --lr 0.001 \
    --batch_size 64 \
    --weight_decay 0.001 \
    --save_dir "/path/to/in9-models/${exp_name}" \
    --device "cuda"
done
