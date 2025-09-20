import torch


class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型配置
    model_id = "runwayml/stable-diffusion-v1-5"
    clip_model_id = "openai/clip-vit-base-patch32"

    # 生成配置
    prompt = "a spaceship landing on a distant planet, science fiction, high resolution"
    negative_prompt = "blurry, low quality, distortion"
    seed = 42
    num_inference_steps = 50

    # 路径配置
    output_dir = "./outputs"
    generated_images_dir = f"{output_dir}/generated_images"
    evaluation_results_dir = f"{output_dir}/evaluation_results"
    training_logs_dir = f"{output_dir}/training_logs"

    # 可视化配置
    visualization_steps = 10  # 每多少步可视化一次