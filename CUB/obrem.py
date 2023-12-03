import PIL
import numpy as np
import torch
from sd_cn_inpaint_pipeline import StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline

from diffusers import ControlNetModel, DEISMultistepScheduler
from PIL import Image
import cv2

controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16,
                                             cache_dir="/fs/nexus-scratch/rezashkv/.cache/huggingface/")

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet,
    torch_dtype=torch.float16,
    cache_dir="/fs/nexus-scratch/rezashkv/.cache/huggingface/"
)

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

# speed up diffusion process with faster scheduler and memory optimization
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')


def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)


def predict(input_dict):
    # Get the drawn input image and mask
    image = input_dict["image"].convert("RGB")
    input_image = input_dict["mask"].convert("RGB")
    # input_image = resize_image(input_image, 768)
    # image = resize_image(image, 768)

    # Convert images to numpy arrays
    image_np = np.array(image)
    input_image_np = np.array(input_image)

    mask_np = np.zeros_like(input_image_np)
    mask_np[input_image_np < 40] = 255
    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY) / 255.0

    mask = Image.fromarray(np.uint8(mask_np * 255))

    # Apply OpenCV inpainting
    inpainted_image_np = cv2.inpaint(input_image_np, (mask_np * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # Blend the original image and the inpainted image using the mask
    blended_image_np = image_np * (1 - mask_np)[:, :, None] + inpainted_image_np * mask_np[:, :, None]

    # Convert the blended image back to a PIL Image
    blended_image = Image.fromarray(np.uint8(blended_image_np))
    blended_image.save("blended.png")

    # Process the blended image
    blended_image_np = np.array(blended_image)
    low_threshold = 10
    high_threshold = 20
    canny = cv2.Canny(blended_image_np, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    canny_image.save("canny.png")

    generator = torch.manual_seed(0)
    output = pipe(
        prompt="",
        num_inference_steps=20,
        generator=generator,
        image=blended_image_np,
        control_image=canny_image,
        controlnet_conditioning_scale=0.9,
        mask_image=mask
    ).images[0]

    return blended_image



