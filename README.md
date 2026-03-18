# 🎨 DCGAN 动漫头像生成项目

## 📌 项目简介

本项目基于 **PyTorch** 实现了深度卷积生成对抗网络（DCGAN），用于生成高质量动漫头像。生成器通过反卷积层将随机噪声映射为 **64×64 RGB 图像**，判别器通过卷积层判断图像的真伪。训练过程中采用 **批量归一化（BatchNorm）**、**标签平滑（Label Smoothing）** 和 **噪声注入** 等技巧来提高训练的稳定性和收敛速度。  

项目展示了完整的 **GAN 训练流程**，包括生成器和判别器的端到端训练、生成图像可视化以及损失曲线绘制，体现了对深度学习和生成对抗网络的实践能力。

---

## 🏗️ 模型结构

### 生成器（Generator）
- **输入**：100维随机噪声向量  
- **结构**：多层 `ConvTranspose2d` 上采样  
- **输出**：64×64 RGB 图像

### 判别器（Discriminator）
- **输入**：64×64 RGB 图像  
- **结构**：多层 `Conv2d` 下采样  
- **输出**：图像真实/生成概率

---

## ⚙️ 训练参数

| 参数 | 值 |
|------|----|
| Epochs | 100 |
| Batch Size | 64 |
| 学习率 | 0.0002 |
| 优化器 | Adam (betas=(0.5,0.999)) |
| 随机噪声维度 (z_dim) | 100 |

---

## 📂 数据集

使用 **Anime Face Dataset**：  
- 图像尺寸统一调整为 **64×64**  
- 数据归一化到 **[-1, 1]**  

[数据目录示例：
/kaggle/input/anime-faces/data](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

---

## 📊 功能特点

- DCGAN 卷积网络训练  
- 自动绘制生成器与判别器损失曲线  
- 每 10 个 Epoch 自动保存生成的中间图片  
- 支持 **GPU 加速训练**  
- 训练完成后保存模型文件 (`generator.pth` / `discriminator.pth`)

---
## ⚡ 快速开始

克隆仓库：  
```bash
git clone https://github.com/你的用户名/anime-dcgan.git
cd anime-dcgan



安装依赖：
```bash
pip install -r requirements.txt


准备数据集：
```bash
anime-DCGAN/datasets/


训练模型：
```bash
python train.py


生成图片：
```bash
python generate.py


---


## 🖼️ 生成效果

训练过程中生成器输出示例：

- **Epoch 10** → 模糊初始图像
<img width="947" height="997" alt="image" src="https://github.com/user-attachments/assets/8f01b143-f6f1-48e5-8a1b-2afcadce35e6" />

- **Epoch 50** → 人脸结构初步形成  
- **Epoch 100** → 清晰动漫人脸  

生成的图像将保存在：
outputs/images/

---

## 🔧 文件结构
anime-dcgan/
│
├── models/
│   ├── generator.py
│   └── discriminator.py
│
├── dataset/
│   └── dataset.py
│
├── train.py
├── generate.py
├── utils.py
│
├── outputs/
│   ├── images/
│   └── models/
│
├── README.md
└── requirements.txt
