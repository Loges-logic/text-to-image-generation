import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from src.utils.config import Config


class ImageGenerator:
    def __init__(self):
        self.config = Config()
        self.pipe = self._load_model()

    def _load_model(self):
        """加载Stable Diffusion模型"""
        scheduler = DDIMScheduler.from_pretrained(self.config.model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        ).to(self.config.device)
        return pipe

    def generate_image(self, prompt=None, negative_prompt=None, seed=None, num_steps=None):
        """生成图像"""
        # 使用配置或参数
        prompt = prompt or self.config.prompt
        negative_prompt = negative_prompt or self.config.negative_prompt
        seed = seed or self.config.seed
        num_steps = num_steps or self.config.num_inference_steps

        # 设置随机种子确保可复现性
        generator = torch.Generator(device=self.config.device).manual_seed(seed)

        print("Generating image...")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_steps
        ).images[0]

        # 保存图像
        os.makedirs(self.config.generated_images_dir, exist_ok=True)
        image_path = os.path.join(self.config.generated_images_dir, f"generated_image_seed_{seed}.png")
        image.save(image_path)
        print(f"Image saved to {image_path}")

        return image, image_path