from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import base64
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2.5-1024px-aesthetic",
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt, num_inference_steps=50, guidance_scale=3).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        return { "generated_image_base64" : img_str }
        
    def finalize(self):
        self.pipe = None
