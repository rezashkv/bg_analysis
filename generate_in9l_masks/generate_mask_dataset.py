import argparse
import os
import datetime
from rembg import remove, new_session
import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument('--category-folder-name', type=str)
args = parser.parse_args()

DATASET_ROOT_PATH = "XXX/in9l/train" # Path to in9l train
OUTPUT_ROOT_PATH = "XXX/in9l_masks/train" # Path to in9_masks train

INPUT_SIZE = 128

dataset_folder_path = os.path.join(DATASET_ROOT_PATH, args.category_folder_name)
assert os.path.exists(dataset_folder_path)

output_folder_path = os.path.join(OUTPUT_ROOT_PATH, args.category_folder_name)
if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

session = new_session()

print("Start -", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
cnt = 0

i_list, o_list = [], []

for file_name in os.listdir(dataset_folder_path):
    input_path = os.path.join(dataset_folder_path, file_name)
    output_path = os.path.join(output_folder_path, file_name)

    i_list.append(input_path)
    o_list.append(output_path)


def work(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, only_mask=True)
            o.write(output)

with mp.Pool(mp.cpu_count() - 2) as pool:
    results = pool.starmap(work, zip(i_list, o_list))

print("Finish -", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
