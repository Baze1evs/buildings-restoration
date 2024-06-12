import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

batch_size = 4
for i in range(250):
    prompt = "a photo of a building with cosmetic damage"
    images = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=batch_size).images

    for j, img in enumerate(images):
        img.save(f"DilapidatedBuildings/{batch_size * i + j}.jpg")
