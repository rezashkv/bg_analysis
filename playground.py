import os

path = "/fs/nexus-scratch/rezashkv/research/data/bg_challenge/test/original/val"

#iterate over all folders in path and print the number of files in each folder
for folder in os.listdir(path):
    print(folder, len(os.listdir(os.path.join(path, folder))))
