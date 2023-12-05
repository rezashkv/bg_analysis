#!/bin/bash
#SBATCH --job-name=in9l_mask_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger

source .venv_cpu/bin/activate
python generate_mask_dataset.py --category-folder-name=XX_YY...Y # Category

