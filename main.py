import os
import warnings

warnings.filterwarnings("ignore", message=".*The converter attribute was deprecated.*")

from src.generation.image_generator import ImageGenerator
from src.evaluation.clip_evaluator import CLIPEvaluator
from src.visualization.denoising_visualizer import DenoisingVisualizer
from src.utils.config import Config


def main():
    # 初始化配置
    config = Config()

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.generated_images_dir, exist_ok=True)
    os.makedirs(config.evaluation_results_dir, exist_ok=True)
    os.makedirs(config.training_logs_dir, exist_ok=True)

    print(f"Using device: {config.device}")

    # 生成图像
    generator = ImageGenerator()
    image, image_path = generator.generate_image()

    # 评估图像
    evaluator = CLIPEvaluator()
    clip_score = evaluator.evaluate_image(image, config.prompt)
    print(f"CLIP Score (higher is better): {clip_score:.4f}")

    # 保存评估结果
    evaluator.save_evaluation_results(
        config.prompt,
        clip_score,
        image_path,
        "CLIP Score measures text-image similarity. Higher values indicate better alignment with the prompt."
    )

    # 可视化去噪过程
    visualizer = DenoisingVisualizer(generator.pipe)
    denoising_path, steps_csv_path = visualizer.visualize_denoising_process(config.prompt)
    print(f"Denoising visualization completed. Check {denoising_path}")

    # 创建README文件
    with open(os.path.join(config.output_dir, "README.md"), "w") as f:
        f.write("# Stable Diffusion Generation Results\n\n")
        f.write("## Project Overview\n")
        f.write("This project demonstrates image generation using Stable Diffusion v1-5.\n\n")
        f.write("## Generated Image\n")
        f.write(f"![Generated Image]({os.path.basename(image_path)})\n\n")
        f.write("## Evaluation Results\n")
        f.write(f"- **Prompt**: {config.prompt}\n")
        f.write(f"- **CLIP Score**: {clip_score:.4f}\n")
        f.write(
            "- **Evaluation Note**: CLIP Score measures text-image similarity. Higher values indicate better alignment with the prompt.\n\n")
        f.write("## Denoising Process\n")
        f.write(f"![Denoising Process]({os.path.basename(denoising_path)})\n")
        f.write(f"Detailed step information available in [denoising_steps.csv]({os.path.basename(steps_csv_path)})\n")

    print("All processes completed successfully!")


if __name__ == "__main__":
    main()