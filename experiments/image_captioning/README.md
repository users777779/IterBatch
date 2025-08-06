# 图像描述实验

## 实验目的
本实验旨在验证ABR策略在视觉语言任务上的有效性，比较不同训练策略对图像描述生成质量的影响。

## 实验设置
- **模型**：BLIP或ViT-GPT2结构的预训练模型 + LoRA微调
- **数据集**：Flickr30k或COCO Captions
- **指标**：BLEU, CIDEr, ROUGE分数

## 实验流程
1. **基线实验**：每个batch只训练一次
2. **ABR策略实验**：根据loss动态调整batch的训练次数
3. **可学习调度实验**：使用神经网络学习最优训练次数

## 目录结构
```
image_captioning/
├── README.md
├── environment.yml  # 环境配置文件
├── model.py        # 模型定义与加载
├── data_loader.py  # 数据加载
├── strategy.py     # 训练策略
├── train.py        # 训练脚本
└── results/        # 实验结果
```

## 环境设置
推荐使用Anaconda创建独立虚拟环境：
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate abr-image-captioning
```

## 运行方法
```bash
cd /root/IterBatch/experiments/image_captioning
# 确保已激活环境
conda activate abr-image-captioning
python train.py --model blip --dataset coco --strategy abr --lora_rank 8
```

## 预期结果
我们期望ABR策略能够在更少的训练步骤内生成质量更高的图像描述，同时提高模型对复杂图像内容的理解能力。