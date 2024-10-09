"""
Beyond images
Diffusion models can also be used for video generation. At the moment of the writing, this field is still in its
infancy, but it is progressing fast so keep an eye on the available models as there might be much better ones
by the time you are reading this.

The list of available model for text-to-video is available here
"""
import torch
from diffusers import DiffusionPipeline

from helpers import get_video
from IPython.display import Video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16,
                                         variant="fp16").to("cuda")

prompt = "Earth sphere from space"

rand_gen = torch.manual_seed(42312981)
frames = pipe(prompt, generator=rand_gen).frames

Video(get_video(frames, "earth.mp4"))
