import os
import torch
from transformers import CLIPProcessor, CLIPModel
from src.utils.config import Config


class CLIPEvaluator:
    def __init__(self):
        self.config = Config()
        self.clip_model, self.clip_processor = self._load_model()

    def _load_model(self):
        """加载CLIP模型"""
        clip_model = CLIPModel.from_pretrained(self.config.clip_model_id).to(self.config.device)
        clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_id)
        return clip_model, clip_processor

    def evaluate_image(self, image, prompt):
        """评估图像与文本的相似度"""
        # 处理图像和文本
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_score = logits_per_image.item()

        return clip_score

    def save_evaluation_results(self, prompt, clip_score, image_path, additional_notes=""):
        """保存评估结果"""
        os.makedirs(self.config.evaluation_results_dir, exist_ok=True)
        result_file = os.path.join(self.config.evaluation_results_dir, "evaluation_results.csv")

        # 如果文件不存在，创建并写入标题
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write("Prompt,CLIP Score,Image Path,Additional Notes\n")

        # 追加结果
        with open(result_file, "a") as f:
            f.write(f'"{prompt}",{clip_score:.4f},{image_path},"{additional_notes}"\n')

        print(f"Evaluation results saved to {result_file}")
        return result_file