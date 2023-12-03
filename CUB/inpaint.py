import PIL
import cv2
import numpy as np
import torch

from diffusers import AutoPipelineForInpainting, KandinskyInpaintCombinedPipeline
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16,
    cache_dir="/fs/nexus-scratch/rezashkv/.cache/huggingface/"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

img_path = "/fs/nexus-scratch/rezashkv/research/data/bg_challenge/CUB_200/bg-only-black/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
mask_path = "/fs/nexus-scratch/rezashkv/research/data/bg_challenge/CUB_200/segmentations/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.png"

# image = load_image(img_path)
# image = np.array(image)
#
# # Define the coordinates of the bounding box (assuming you have them)
# # Replace these values with the coordinates obtained from your object detection algorithm
# x1, y1, x2, y2 = 50, 50, 280, 200  # Example coordinates
#
# Create a mask for the black box within the bounding box
mask = np.zeros_like(image)
cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Create a white rectangle

# Inpainting - fill the black box with information from the surrounding area
inpaint = cv2.inpaint(image, mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_TELEA)
#
# # Save the inpainted image
# cv2.imwrite("inpaint.png", inpaint)

# init_image = load_image(img_path).resize((512, 512))
# mask_image = load_image(mask_path).resize((512, 512))
#
# # mask_image = np.array(mask_image)
# mask_image = np.zeros_like(init_image)
# mask_image[init_image == 0] = 255
# mask_image = PIL.Image.fromarray(mask_image)
#
# generator = torch.Generator("cuda").manual_seed(92)
# prompt = "background"
# negative_prompt = ""
# repainted_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, negative_prompt=negative_prompt).images[0]
# repainted_image.save("repainted_image.png")

# # Convert mask to grayscale NumPy array
# mask_image_arr = np.array(mask_image.convert("L"))
# # Add a channel dimension to the end of the grayscale mask
# mask_image_arr = mask_image_arr[:, :, None]
# # Binarize the mask: 1s correspond to the pixels which are repainted
# mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
# mask_image_arr[mask_image_arr < 0.5] = 0
# mask_image_arr[mask_image_arr >= 0.5] = 1
#
# # Take the masked pixels from the repainted image and the unmasked pixels from the initial image
# unmasked_unchanged_image_arr = (1 - mask_image_arr) * init_image + mask_image_arr * repainted_image
# unmasked_unchanged_image = PIL.Image.fromarray(unmasked_unchanged_image_arr.round().astype("uint8"))
# unmasked_unchanged_image.save("force_unmasked_unchanged.png")
# make_image_grid([init_image, mask_image, repainted_image, unmasked_unchanged_image], rows=2, cols=2)