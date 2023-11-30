#!/bin/bash
#SBATCH --job-name=in9-bg-reliance                        # Specify a name for your job
#SBATCH --output=slurm-logs/out-in9-bg-reliance.log       # Specify the output log file
#SBATCH --error=slurm-logs/err-in9-bg-reliance.log        # Specify the error log file
#SBATCH --nodes=1                                         # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00                                   # Maximum execution time (HH:MM:SS)
                                                          # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1                                 # Number of CPU cores per task
#SBATCH --partition=scavenger                             # Specify the partition (queue) you want to use
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:4                             # Number of GPUs per node
#SBATCH --mem=32G                                         # Memory per node

dataset_variations=("mixed_rand")
exp_name="bg-reliance"
dataset="data/in9"
proj_root="/vulcanscratch/mhoover4/code/bg_analysis"

# Load any required modules or activate your base environment here if necessary
source "${proj_root}/.venv/bin/activate"

cd ${proj_root}/imagenet || exit

for dataset_v in "${dataset_variations[@]}"; do
  accelerate launch --multi_gpu  bg_reliance_script.py \
  --train_dir "${proj_root}/${dataset}/${dataset_v}/train" \
  --validation_dir "${proj_root}/${dataset}/${dataset_v}/val" \
  --test_dir "${proj_root}/${dataset}/test/${dataset_v}/val" \
  --original_test_dir "${proj_root}/${dataset}/test/original/val" \
  --output_dir "./outputs-${dataset}-bg-reliance-${dataset_v}" \
  --cache_dir "/vulcanscratch/mhoover4/.cache/bg_reliance" \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --num_train_epochs 100 \
  --per_device_train_batch_size 128 \
  --with_tracking \
  --experiment_name "${exp_name}-${dataset_v}" \
  --lr_scheduler_type linear
done
