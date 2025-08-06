# 图像分类实验

## 实验目的
本实验旨在验证ABR策略在图像分类任务上的有效性，比较不同训练策略对模型性能和收敛速度的影响。

## 实验设置
- **模型**：ResNet-18/Transformer/CNN
- **数据集**：MNIST/CIFAR-10/ImageNet
- **指标**：准确率(accuracy)、损失值(loss)

## 实验流程
1. **基线实验**：每个batch只训练一次
2. **ABR策略实验**：根据loss动态调整batch的训练次数
3. **可学习调度实验**：使用神经网络学习最优训练次数

## 目录结构
```
image_classification/
├── README.md
├── environment.yml  # 环境配置文件
├── model.py        # 模型定义
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
conda activate abr-image-classification
```

## 运行方法
```bash
cd /root/IterBatch/experiments/image_classification
# 确保已激活环境
conda activate abr-image-classification
python train.py --model resnet18 --dataset cifar10 --strategy abr
```

## 预期结果
我们期望ABR策略能够在更少的训练步骤内达到与基线相当或更好的准确率，同时提高模型对困难样本的学习能力。