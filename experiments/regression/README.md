# 回归任务实验

## 实验目的
本实验旨在验证ABR策略在回归任务上的有效性，比较不同训练策略对模型性能和收敛速度的影响。

## 实验设置
- **模型**：MLP (多层感知器)
- **数据集**：加州房价预测(California Housing)或波士顿房价(Boston Housing)数据集
- **指标**：均方根误差(RMSE)、平均绝对误差(MAE)
- **batch_size**：5

## 实验流程
1. **基线实验**：每个batch只训练一次
2. **ABR策略实验**：根据loss动态调整batch的训练次数
3. **可学习调度实验**：使用神经网络学习最优训练次数

## 目录结构
```
regression/
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
conda activate abr-regression
```

## 运行方法
```bash
cd /root/IterBatch/experiments/regression
# 确保已激活环境
conda activate abr-regression
python train.py --dataset california_housing --strategy abr
```

## 预期结果
我们期望ABR策略能够在更少的训练步骤内达到与基线相当或更好的回归性能，同时提高模型对异常样本的拟合能力。