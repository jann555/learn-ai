from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid
from diffusers import KandinskyV22Img2ImgPipeline, KandinskyPriorPipeline
import numpy as np

import PIL
import torch

image = PIL.Image.open("sketch.png")
'''Now we can use the [Kandinsky](https://github.com/ai-forever/Kandinsky-2) model to generate an image that 
respects the subjects and their positions in our sketch:'''

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = KandinskyV22Img2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

original_image = image.copy().resize((768, 768))

prompt = "A photograph of a house in the fall, high details, broad daylight"
negative_prompt = "low quality, bad quality"

rand_gen = torch.manual_seed(67806801)
image_embeds, negative_image_embeds = prior_pipeline(prompt, negative_prompt, generator=rand_gen).to_tuple()

new_image = pipeline(
    image=original_image,
    image_embeds=image_embeds,
    negative_image_embeds=negative_image_embeds,
    height=768,
    width=768,
    strength=0.35,
    generator=rand_gen
).images[0]
fig = make_image_grid([original_image.resize((512, 512)), new_image.resize((512, 512))], rows=1, cols=2)
fig.save('Realistic_sketch.png')