# create some variations of the dataset
# 1. create a dataset with only the foreground based on the segmentation mask
# 2. create a dataset with no foreground based on the bounding box
# 3. create a dataset with only the background but the foreground is inpainted
# 4. create a dataset with only the foreground
# 5. create a dataset with the background replaced by a random background
# 6. create a dataset with the background replaced by a background from the same class

import argparse
import os
import torchvision
import numpy as np
import torch
from sd_cn_inpaint_pipeline import StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline

from diffusers import ControlNetModel, DEISMultistepScheduler
from PIL import Image
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True,)
parser.add_argument('--segmentation_dir', type=str, required=True)
parser.add_argument('--bg_only_inpaint_dir', type=str, required=True)
parser.add_argument('--fg_only_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_classes', type=int, default=200)


args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16,
#                                              cache_dir="/fs/nexus-scratch/rezashkv/.cache/huggingface/")
#
# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet,
#     torch_dtype=torch.float16,
#     cache_dir="/fs/nexus-scratch/rezashkv/.cache/huggingface/"
# )
#
# pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
#
# # speed up diffusion process with faster scheduler and memory optimization
# # remove following line if xformers is not installed
# pipe.enable_xformers_memory_efficient_attention()
# pipe.to('cuda')


def predict(input_dict):
    # Get the drawn input image and mask
    image = input_dict["image"].convert("RGB")
    input_image = input_dict["mask"].convert("RGB")

    # Convert images to numpy arrays
    image_np = np.array(image)
    input_image_np = np.array(input_image)

    mask_np = np.zeros_like(input_image_np)
    mask_np[input_image_np < 40] = 255
    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY) / 255.0

    # Apply OpenCV inpainting
    inpainted_image_np = cv2.inpaint(input_image_np, (mask_np * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # Blend the original image and the inpainted image using the mask
    blended_image_np = image_np * (1 - mask_np)[:, :, None] + inpainted_image_np * mask_np[:, :, None]

    # Convert the blended image back to a PIL Image
    blended_image = Image.fromarray(np.uint8(blended_image_np))
    return blended_image


dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, 'images'))
segmentation_dataset = torchvision.datasets.ImageFolder(args.segmentation_dir)
bounding_boxes = np.loadtxt(os.path.join(args.dataset_dir, 'bounding_boxes.txt'), dtype=np.float32)
bg_only_inpaint_dataset = torchvision.datasets.ImageFolder(args.bg_only_inpaint_dir)
fg_only_dataset = torchvision.datasets.ImageFolder(args.fg_only_dir)
image_bb_map = {}
with open(os.path.join(args.dataset_dir, 'images.txt')) as f:
    for line in f:
        img_id, img_name = line.strip().split()
        image_bb_map[img_name] = int(img_id) - 1

for i in range(len(dataset)):
    # select a random index from the dataset from the same class not including the current index
    j = np.random.choice(np.where(np.array(dataset.targets) == dataset.targets[i])[0][np.where(np.array(dataset.targets) == dataset.targets[i])[0] != i])
    # select a random index from the dataset from a different class
    k = np.random.choice(np.where(np.array(dataset.targets) != dataset.targets[i])[0])
    # select a random index from the dataset from the next class
    l = np.random.choice(np.where(np.array(dataset.targets) == (dataset.targets[i] + 1) % args.num_classes)[0])

    img, label = dataset[i]
    img_path = dataset.imgs[i][0].split('/')
    img_dir, img_name = img_path[-2], img_path[-1]
    # img = torch.tensor(np.array(img))
    img = np.array(img)

    bg_j = bg_only_inpaint_dataset[j][0]
    bg_k = bg_only_inpaint_dataset[k][0]
    bg_l = bg_only_inpaint_dataset[l][0]
    # resize the backgrounds to the same size as the image
    bg_j = cv2.resize(np.array(bg_j), (img.shape[1], img.shape[0]))
    bg_k = cv2.resize(np.array(bg_k), (img.shape[1], img.shape[0]))
    bg_l = cv2.resize(np.array(bg_l), (img.shape[1], img.shape[0]))

    foreground_img, _ = fg_only_dataset[i]
    foreground_img = np.array(foreground_img)

    # put the foreground image in each of the backgrounds
    bg_j[foreground_img > 5] = img[foreground_img > 5]
    bg_k[foreground_img > 5] = img[foreground_img > 5]
    bg_l[foreground_img > 5] = img[foreground_img > 5]

    bg_j = torch.tensor(bg_j)
    bg_k = torch.tensor(bg_k)
    bg_l = torch.tensor(bg_l)

    bg_j = bg_j.permute(2, 0, 1).float() / 255.0
    bg_k = bg_k.permute(2, 0, 1).float() / 255.0
    bg_l = bg_l.permute(2, 0, 1).float() / 255.0

    # segmentation_img, _ = segmentation_dataset[i]

    # segmentation_img = torch.tensor(np.array(segmentation_img))
    # bb = bounding_boxes[image_bb_map[img_dir + '/' + img_name]].astype(np.int32)[1:]
    # bb is x, y, width, height and we want x1, y1, x2, y2
    # bb[2:] += bb[:2]

    # foreground_no_fg = torch.zeros_like(img)
    # foreground_fg_only = torch.zeros_like(img)
    # bg_only_black = np.zeros_like(img)
    # bg_only_sd = torch.zeros_like(img)

    # foreground_fg_only[segmentation_img > 128] = img[segmentation_img > 128]
    # foreground_no_fg[segmentation_img == 0] = img[segmentation_img == 0]

    # fill the foreground outside the bounding box with the image
    # bg_only_black[:bb[1], :, :] = img[:bb[1], :, :]
    # bg_only_black[bb[3]:, :, :] = img[bb[3]:, :, :]
    # bg_only_black[:, :bb[0], :] = img[:, :bb[0], :]
    # bg_only_black[:, bb[2]:, :] = img[:, bb[2]:, :]

    # get the largest rectangle in the image outside the bounding box
    # bg_only_sd = predict({'image': img, 'mask': segmentation_img})
    # x1, y1, x2, y2 = bb
    # mask = np.zeros_like(img)
    # cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Create a white rectangle

    # Inpainting - fill the black box with information from the surrounding area
    # inpaint = cv2.inpaint(bg_only_black, mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # bg_only_inpainted = torch.tensor(np.array(inpaint))

    # foreground_fg_only = foreground_fg_only.permute(2, 0, 1).float() / 255.0
    # foreground_no_fg = foreground_no_fg.permute(2, 0, 1).float() / 255.0
    # bg_only_black = bg_only_black.permute(2, 0, 1).float() / 255.0
    # bg_only_inpainted = bg_only_inpainted.permute(2, 0, 1).float() / 255.0

    # os.makedirs(os.path.join(args.save_dir, 'fg-only', img_dir), exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, 'no-fg', img_dir), exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, 'bg-only-black', img_dir), exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, 'bg-only-inpaint', img_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'mixed-same', img_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'mixed-rand', img_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'mixed-next', img_dir), exist_ok=True)

    torchvision.utils.save_image(bg_j, os.path.join(args.save_dir, 'mixed-same', img_dir, img_name))
    torchvision.utils.save_image(bg_k, os.path.join(args.save_dir, 'mixed-rand', img_dir, img_name))
    torchvision.utils.save_image(bg_l, os.path.join(args.save_dir, 'mixed-next', img_dir, img_name))

    # torchvision.utils.save_image(foreground_fg_only, os.path.join(args.save_dir, 'fg-only', img_dir, img_name))
    # torchvision.utils.save_image(foreground_no_fg, os.path.join(args.save_dir, 'no-fg', img_dir, img_name))
    # torchvision.utils.save_image(bg_only_black, os.path.join(args.save_dir, 'bg-only-black', img_dir, img_name))
    # torchvision.utils.save_image(bg_only_inpainted, os.path.join(args.save_dir, 'bg-only-inpaint', img_dir, img_name))



