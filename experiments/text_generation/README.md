# 文本生成实验

## 实验目的
本实验旨在验证ABR策略在生成式语言任务上的有效性，比较不同训练策略对文本生成质量和模型效率的影响。

## 实验设置
- **模型**：LLaMA + LoRa
- **数据集**：WikiText-103或一个对话数据集
- **指标**：困惑度(Perplexity, PPL)（越低越好）

## 实验流程
1. **基线实验**：每个batch只训练一次
2. **ABR策略实验**：根据loss动态调整batch的训练次数
3. **可学习调度实验**：使用神经网络学习最优训练次数

## 目录结构
```
text_generation/
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
conda activate abr-text-generation
```

## 运行方法
```bash
cd /root/IterBatch/experiments/text_generation
# 确保已激活环境
conda activate abr-text-generation
python train.py --model llama --dataset wikitext103 --strategy abr --lora_rank 8
```

## 预期结果
我们期望ABR策略能够降低模型的困惑度，生成更流畅、更符合上下文的文本，同时减少训练所需的计算资源。