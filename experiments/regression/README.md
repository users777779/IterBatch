# 回归任务实验

## 项目结构

- `train.py`: 原始的单一策略训练脚本
- `train_comparison.py`: 多策略对比实验脚本
- `train.py`: 新的统一训练脚本（支持所有策略）
- `model.py`: 模型定义
- `strategy.py`: 训练策略实现
- `data_loader.py`: 数据加载器

## 策略实现原理

### 1. 基线策略 (Baseline)

最简单的训练策略，每个批次数据只进行一次前向传播和反向传播，用于与其他高级策略进行对比。

### 2. ABR策略 (Adaptive Batch Repetition)

根据每个批次的初始损失值动态决定该批次的重复训练次数。损失值高的难训练批次会被训练更多次，损失值低的易训练批次则训练次数较少。

### 3. 可学习调度策略 (Learnable Scheduling)

使用一个额外的调度模型来学习如何根据批次的初始损失值预测最优的重复训练次数。调度模型会根据训练过程中的反馈进行优化。

### 4. 滑动窗口策略 (Sliding Window)

维护一个固定大小的滑动窗口，记录最近几个批次的损失值，并根据损失的变化趋势来调整当前批次的重复训练次数。如果损失呈上升趋势且当前损失高于阈值，则增加重复训练次数。

## 评估指标说明

在回归任务中，我们使用以下指标来评估模型性能：

- **Loss (MSE)**: 均方误差，衡量预测值与真实值之间的平均平方差
- **RMSE**: 均方根误差，MSE的平方根，与目标变量具有相同的单位
- **MAE**: 平均绝对误差，衡量预测值与真实值之间的平均绝对差
- **Accuracy**: 准确率，定义为预测值与真实值之差小于0.1的比例

## 使用方法

### 使用统一训练脚本

新的统一训练脚本 `train.py` 支持所有四种策略的训练：

```bash
# 训练单一策略
python train.py --strategies baseline

# 训练多种策略
python train.py --strategies baseline abr learnable window

# 指定数据集和其他参数
python train.py --strategies baseline abr --dataset california --batch_size 32 --epochs 10
```

### 参数说明

- `--strategies`: 要训练的策略，可选值包括 `baseline`, `abr`, `learnable`, `window`
- `--dataset`: 数据集选择，可选 `california` 或 `boston`，默认为 `california`
- `--batch_size`: 批次大小，默认为 32
- `--epochs`: 训练轮数，默认为 10
- `--lr`: 学习率，默认为 0.001
- `--hidden_dim`: 隐藏层维度，默认为 64
- `--loss_threshold`: ABR策略的损失阈值，默认为 0.3
- `--max_repeats`: ABR策略的最大重复次数，默认为 5
- `--window_size`: 滑动窗口策略的窗口大小，默认为 5
- `--save_dir`: 结果保存目录，默认为 `results`

## 查看结果

训练结果会保存在指定的保存目录中，包括：
1. 每种策略的指标历史文件（`.npy`格式）
2. TensorBoard日志文件

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir=results
```

然后在浏览器中打开 `http://localhost:6006` 查看训练指标。