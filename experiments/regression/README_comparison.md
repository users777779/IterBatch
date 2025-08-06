# 回归任务对比实验

## 实验目的
本实验旨在对比三种不同训练策略在回归任务上的性能差异：
1. 基线策略：每个批次只训练一次
2. ABR策略：根据损失动态调整批次的重复训练次数（最多重复1次）
3. 可学习调度策略：使用神经网络学习最优训练次数

## 实验设计

本对比实验将同时训练四种不同策略的模型：
1. **基线策略**：每个批次仅训练一次
2. **ABR策略**：根据初始损失动态调整重复训练次数
3. **可学习调度策略**：使用神经网络预测最优重复训练次数
4. **滑动窗口策略**：通过固定大小窗口监控最近批次损失变化趋势，动态调整重复训练次数

### 模型与数据
- **模型**：MLP (多层感知器)
- **数据集**：加州房价预测(California Housing)或波士顿房价(Boston Housing)数据集
- **指标**：均方根误差(RMSE)、平均绝对误差(MAE)
- **batch_size**：5

### 关键设计点
- **权重一致性**：只创建一个主模型，然后复制给四个实验模型，确保初始权重完全一致
- **数据一致性**：四个模型同时训练同一个batch，确保数据输入完全一致
- **训练轮次**：固定为3轮epoch
- **重复训练限制**：所有策略最多重复训练1次
- **滑动窗口大小**：默认5个批次

## 目录结构
```
regression/
├── README.md               # 原始实验文档
├── README_comparison.md    # 对比实验文档
├── environment.yml         # 环境配置文件
├── model.py                # 模型定义
├── data_loader.py          # 数据加载
├── strategy.py             # 训练策略
├── train.py                # 原始训练脚本
├── train_comparison.py     # 对比实验训练脚本
└── results_comparison/     # 对比实验结果
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
python train_comparison.py --dataset california_housing
```

### 可选参数
- `--dataset`: 选择数据集 (california/boston)，默认为california
- `--batch_size`: 批次大小，默认为5
- `--epochs`: 训练轮数，默认为3
- `--lr`: 学习率，默认为0.001
- `--hidden_dim`: 隐藏层维度，默认为64
- `--loss_threshold`: ABR策略和滑动窗口策略的损失阈值，默认为0.5
- `--max_repeats`: 所有策略的最大重复次数，默认为1
- `--window_size`: 滑动窗口策略的窗口大小，默认为5
- `--save_dir`: 结果保存目录，默认为results_comparison

## 预期结果
运行后将在结果保存目录下生成：
1. 四个策略的指标历史数据（.npy文件）
2. 损失曲线对比图 (loss_comparison.png)
3. RMSE曲线对比图 (rmse_comparison.png)
4. MAE曲线对比图 (mae_comparison.png)
5. 滑动窗口策略的损失趋势变化图 (window_trend.png)

## 结果解释
- 训练过程中会实时打印每个epoch的训练和测试指标
- 图表将直观展示四种策略在损失、RMSE和MAE上的差异
- 预计**ABR策略**在更少的训练步骤内达到与基线策略相当或更好的性能
- 预计**可学习调度策略**能够自适应不同样本的难度，进一步提高模型性能
- 预计**滑动窗口策略**能够捕捉到损失的变化趋势，在损失上升时增加重复训练，从而提高模型的稳定性
- 我们期望ABR策略、可学习策略和滑动窗口策略能够在相同训练轮次内取得比基线策略更好的性能

## 注意事项
- 确保已安装所有依赖项
- 首次运行可能需要下载数据集，请确保网络连接正常
- 如需调整实验参数，请修改命令行参数或脚本代码