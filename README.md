# bg_analysis
Contextual Bias Analysis for Image Classification Models

## Instructions for training a VIT model on IN9 subsets:

1. Clone this repo.
2. Download test data to `bg_analyis/in9` and rename to `test`:
   ```sh
   cd in9
   wget https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz
   tar -xzf backgrounds_challenge_data.tar.gz
   mv bg_challenge test
   ```
3. Download specific training data subset from https://github.com/MadryLab/backgrounds_challenge#training-data. No renaming is necessary. E.g.:
   ```sh
   wget https://www.dropbox.com/s/cto15ceadgraur2/mixed_rand.tar.gz
   tar -xzf mixed_rand.tar.gz
   ```
4. Install dependencies:
    ```sh
    pip install -r imagenet/requirements.txt
    ```
5. Update `scripts/run-bg-reliance-in9.sh` so that the line `dataset_variations=("only_bg_b" "only_bg_t" "no_fg")` reflects the choice of training dataset.
6. Also update all the paths in the script to point to your project.
7. Run the training with `sbatch scripts/run-bg-reliance-in9.sh`
