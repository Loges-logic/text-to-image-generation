# text-to-image-generation

## 项目描述
本项目使用Stable Diffusion v1-5模型生成图像，并通过CLIP模型评估生成图像与文本提示的相似度。

## 项目结构
stable-diffusion-project/

├── src/ # 源代码目录

│ ├── generation/ # 图像生成模块

│ ├── evaluation/ # 图像评估模块

│ ├── visualization/ # 可视化模块

│ └── utils/ # 工具函数和配置

├── outputs/ # 输出目录

│ ├── generated_images/ # 生成的图像

│ ├── evaluation_results/ # 评估结果

│ └── training_logs/ # 训练日志和可视化

├── requirements.txt # 项目依赖

├── main.py # 主程序入口

└── README.md # 项目说明

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+(如使用GPU)
## 安装依赖
bash

pip install -r requirements.txt
## 使用方法
在src/utils/conkig.py文件中，更改第13行：

prompt = "a spaceship landing on a distant planet, science fiction, high resolution"

更改输入关键词，以生成不同的图像。

运行主程序：

bash

python main.py
## 可复现性
- 固定随机种子(42)确保结果可复现
- requirements.txt明确所有依赖及其版本

## 输出结果
程序将生成：
1. 生成的图像 (outputs/generated_images/)
2. 评估结果CSV文件 (outputs/evaluation_results/)
3. 去噪过程可视化 (outputs/training_logs/)

4. 项目说明文档 (outputs/README.md)
