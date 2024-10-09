'''In the image-to-image task we condition the production of the Diffusion Model through an input image.
There are many ways of doing this. Here we look at transforming a barebone sketch of a scene in a beautiful,
highly-detailed representation of the same.'''

from diffusers import AutoPipelineForText2Image

import torch

prompt = "A tree and a house, made by a child with 3 colors"
rand_gen = torch.manual_seed(41342121)

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

image  = pipe(
    prompt=prompt,
    num_inference_steps=2,
    guidance_scale=2,
    generator=rand_gen
).images[0]

image.save("sketch.png")