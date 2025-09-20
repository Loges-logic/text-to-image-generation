import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.config import Config


class DenoisingVisualizer:
    def __init__(self, pipe):
        self.config = Config()
        self.pipe = pipe

    def visualize_denoising_process(self, prompt, num_steps=None):
        """可视化去噪过程"""
        num_steps = num_steps or self.config.num_inference_steps

        # 重置管道以捕获中间步骤
        self.pipe.scheduler.set_timesteps(num_steps)

        # 设置随机种子确保可复现性
        generator = torch.Generator(device=self.config.device).manual_seed(self.config.seed)
        latents = torch.randn((1, 4, 64, 64), generator=generator, device=self.config.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # 对文本提示进行编码
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to(self.pipe.device))[0]

        # 存储中间图像
        images = []
        step_numbers = []

        for i, t in tqdm(enumerate(self.pipe.scheduler.timesteps), total=num_steps, desc="Denoising"):
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.pipe.unet(latents, t, encoder_hidden_states=text_embeddings).sample

            # 更新latents
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # 定期解码latent为图像
            if i % (num_steps // self.config.visualization_steps) == 0 or i == num_steps - 1:
                latents_decoded = 1 / 0.18215 * latents
                with torch.no_grad():
                    image_step = self.pipe.vae.decode(latents_decoded).sample
                image_step = (image_step / 2 + 0.5).clamp(0, 1)
                image_step = image_step.cpu().permute(0, 2, 3, 1).numpy()[0]
                image_step = (image_step * 255).astype(np.uint8)
                images.append(Image.fromarray(image_step))
                step_numbers.append(i)

        # 保存逐步去噪过程
        os.makedirs(self.config.training_logs_dir, exist_ok=True)

        # 保存为网格图像
        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
        if len(images) == 1:
            axes = [axes]
        for idx, img in enumerate(images):
            axes[idx].imshow(img)
            axes[idx].set_title(f"Step {step_numbers[idx]}")
            axes[idx].axis('off')
        plt.suptitle("Step-by-Step Denoising Process of Stable Diffusion")

        denoising_path = os.path.join(self.config.training_logs_dir, "denoising_process.png")
        plt.savefig(denoising_path)
        plt.close()

        # 保存每个步骤的单独图像
        for idx, img in enumerate(images):
            step_path = os.path.join(self.config.training_logs_dir, f"denoising_step_{step_numbers[idx]}.png")
            img.save(step_path)

        # 保存步骤信息到CSV
        steps_csv_path = os.path.join(self.config.training_logs_dir, "denoising_steps.csv")
        with open(steps_csv_path, "w") as f:
            f.write("Step Number,Image Path\n")
            for idx, step in enumerate(step_numbers):
                f.write(f"{step},{os.path.join(self.config.training_logs_dir, f'denoising_step_{step}.png')}\n")

        print(f"Denoising visualization saved to {denoising_path}")
        return denoising_path, steps_csv_path