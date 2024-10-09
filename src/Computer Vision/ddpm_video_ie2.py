import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from helpers import get_video
from IPython.display import Video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
# These two settings lower the VRAM usage
pipe.enable_model_cpu_offload()
# pipe.unet.enable_forward_chunking()

# Load the conditioning image
image = load_image("in_the_desert_outpaint.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(999)
res = pipe(
    image,
    decode_chunk_size=2,
    generator=generator,
    num_inference_steps=15,
    num_videos_per_prompt=1
)

Video(get_video(res.frames[0], "horse2.mp4"))

image = load_image("xwing.jpeg")
image = image.resize((1024, 576))

generator = torch.manual_seed(999)
res = pipe(
    image,
    decode_chunk_size=2,
    generator=generator,
    num_inference_steps=25,
    num_videos_per_prompt=1
)

Video(get_video(res.frames[0], "xwing.mp4"))
