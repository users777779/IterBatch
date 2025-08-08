# 文本生成实验

## 实验目的
本实验旨在验证ABR策略在生成式语言任务上的有效性，比较不同训练策略对文本生成质量和模型效率的影响。

## 实验设置
- **模型**：LLaMA + LoRA
- **数据集**：WikiText-103或一个对话数据集
- **指标**：困惑度(Perplexity, PPL)（越低越好）

## 实验策略

### 1. 基线策略 (Baseline)
最简单的训练策略，每个批次数据只进行一次前向传播和反向传播，用于与其他高级策略进行对比。

实现细节：
- 继承自 `BaseTrainingStrategy` 基类
- 在 `train_batch` 方法中，对每个批次数据执行标准的前向传播、计算损失、反向传播和优化器更新步骤
- 每个批次固定训练1次，不进行重复训练

### 2. ABR策略 (Adaptive Batch Repetition)
根据每个批次的初始损失值动态决定该批次的重复训练次数。损失值高的难训练批次会被训练更多次，损失值低的易训练批次则训练次数较少。

实现细节：
- 继承自 `BaseTrainingStrategy` 基类
- 在 `train_batch` 方法中，首先进行一次前向传播计算初始损失
- 根据初始损失值与预设阈值的比较，动态确定重复训练次数：
  - 如果初始损失 > 阈值，则重复次数 = min(int(初始损失 / 阈值), 最大重复次数)
  - 否则重复次数 = 1
- 对同一批次数据重复执行指定次数的训练步骤

### 3. 可学习调度策略 (Learnable Scheduling)
使用一个额外的调度模型来学习如何根据批次的初始损失值和困惑度预测最优的重复训练次数。调度模型会根据训练过程中的反馈进行优化。

实现细节：
- 继承自 `BaseTrainingStrategy` 基类
- 使用一个额外的调度模型（`SchedulerLLaMA`）来预测重复训练次数
- 调度模型是一个简单的全连接神经网络，输入为初始损失值和困惑度，输出为预测的重复次数
- 在每次批次训练时：
  - 首先计算初始损失和困惑度
  - 使用调度模型预测重复次数（限制在1-5次之间）
  - 对同一批次数据重复执行预测次数的训练步骤
  - 更新调度模型参数，使其预测更准确

### 4. 滑动窗口策略 (Sliding Window)
维护一个固定大小的滑动窗口，记录最近几个批次的损失值，并根据损失的变化趋势来调整当前批次的重复训练次数。

实现细节：
- 继承自 `BaseTrainingStrategy` 基类
- 使用一个固定大小的滑动窗口（`deque`）存储最近批次的损失值
- 在每次批次训练时：
  - 首先计算初始损失
  - 分析滑动窗口中损失值的变化趋势（通过线性回归计算斜率）
  - 根据趋势、当前损失值和波动性综合决定重复训练次数：
    - 上升趋势且损失高：增加重复次数
    - 下降趋势且损失低：减少重复次数
    - 其他情况：根据损失值和波动性适度调整
  - 对同一批次数据重复执行计算出的次数的训练步骤
  - 将当前批次的初始损失添加到滑动窗口中

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
cd D:\Personal\Desktop\【科研实验】\IterBatch\experiments\text_generation
# 确保已激活环境
conda activate abr-text-generation

# 运行基线策略实验
python train.py --model llama --model_name huggyllama/llama-7b --dataset wikitext --strategies baseline --epochs 3 --batch_size 4

# 运行ABR策略实验
python train.py --model llama --model_name huggyllama/llama-7b --dataset wikitext --strategies abr --epochs 3 --batch_size 4 --loss_threshold 2.0 --max_repeats 5

# 运行可学习调度策略实验
python train.py --model llama --model_name huggyllama/llama-7b --dataset wikitext --strategies learnable --epochs 3 --batch_size 4

# 运行滑动窗口策略实验
python train.py --model llama --model_name huggyllama/llama-7b --dataset wikitext --strategies window --epochs 3 --batch_size 4 --window_size 5 --loss_threshold 2.0 --max_repeats 1

# 运行所有策略实验进行对比
python train.py --model llama --model_name huggyllama/llama-7b --dataset wikitext --strategies baseline abr learnable window --epochs 3 --batch_size 4
```

## 预期结果
我们期望ABR策略能够降低模型的困惑度，生成更流畅、更符合上下文的文本，同时减少训练所需的计算资源。特别是：

1. **基线策略**：作为对比基准，提供标准的训练效果。
2. **ABR策略**：通过动态调整训练次数，有望在保持或提升生成质量的同时减少计算资源消耗。
3. **可学习调度策略**：通过学习最优训练次数，可能实现更精细的训练控制。
4. **滑动窗口策略**：通过分析损失趋势，有望在训练过程中更好地适应数据分布的变化。