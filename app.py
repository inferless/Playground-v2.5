import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

from io import BytesIO
import base64
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
      self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16).to("cuda")
      self.decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16).to("cuda")

    def infer(self, inputs):
      prompt = inputs["prompt"]
      negative_prompt = inputs["negative_prompt"]
      prior_output = self.prior(
          prompt=prompt,
          height=1024,
          width=1024,
          negative_prompt=negative_prompt,
          guidance_scale=4.0,
          num_images_per_prompt=1,
          num_inference_steps=20)

      decoder_output = self.decoder(
          image_embeddings=prior_output.image_embeddings.to(torch.float16),
          prompt=prompt,
          negative_prompt=negative_prompt,
          guidance_scale=0.0,
          output_type="pil",
          num_inference_steps=10
      ).images[0]
      buff = BytesIO()
      decoder_output.save(buff, format="JPEG")
      img_str = base64.b64encode(buff.getvalue()).decode()
      return { "generated_image_base64" : img_str }

    def finalize(self):
        self.pipe = None
