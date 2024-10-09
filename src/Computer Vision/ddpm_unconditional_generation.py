from diffusers import DiffusionPipeline

import torch

'''Unconditional generation
In this type of image generation, the generator is free to do whatever it wants, i.e., 
it is in "slot machine" mode: we pull the lever and the model will generate something related to the training set 
it has been trained on. The only control we have here is the random seed.'''

rand_gen = torch.manual_seed(12418351)

model_name = 'google/ddpm-celebahq-256'

model = DiffusionPipeline.from_pretrained(model_name).to("cuda")
image = model(generator=rand_gen).images[0]
print(image)