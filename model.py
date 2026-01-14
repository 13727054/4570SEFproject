import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    return pipe
