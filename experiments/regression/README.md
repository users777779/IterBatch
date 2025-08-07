# 回归任务实验与性能分析

## 概述
本项目旨在研究和比较不同训练策略在回归任务上的有效性，包括基线策略、ABR策略、可学习调度策略和滑动窗口策略。通过实验，我们分析各种策略对模型性能、收敛速度和训练效率的影响。

## 实验目的
1. 验证ABR策略在回归任务上的有效性
2. 比较不同训练策略（基线、ABR、可学习调度、滑动窗口）的性能差异
3. 分析各策略的训练效率和自适应特性
4. 提供TensorBoard集成方案，方便监控训练过程

## 环境设置
### Anaconda环境配置
推荐使用Anaconda创建独立虚拟环境：
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate abr-regression
```

### 依赖包
环境包含以下主要依赖：
- python=3.9
- pytorch=1.12.1
- torchvision=0.13.1
- scikit-learn=1.1.2
- numpy=1.23.3
- matplotlib=3.6.0
- pandas=2.3.1
- tqdm=4.67.1
- tensorboard=2.12.0 (用于可视化训练过程)

## 实验设计
### 模型与数据
- **模型**：MLP (多层感知器)
- **数据集**：加州房价预测(California Housing)或波士顿房价(Boston Housing)数据集
- **指标**：均方根误差(RMSE)、平均绝对误差(MAE)

### 训练策略
本实验支持四种训练策略：
1. **基线策略**：每个批次仅训练一次
2. **ABR策略**：根据初始损失动态调整重复训练次数
3. **可学习调度策略**：使用神经网络预测最优重复训练次数
4. **滑动窗口策略**：通过固定大小窗口监控最近批次损失变化趋势，动态调整重复训练次数

### 关键设计点
- **权重一致性**：只创建一个主模型，然后复制给各实验模型，确保初始权重一致
- **数据一致性**：各模型同时训练同一个batch，确保数据输入一致
- **重复训练限制**：所有策略最多重复训练次数可配置
- **滑动窗口大小**：默认5个批次

## 目录结构
```
regression/
├── README.md               # 整合实验文档
├── environment.yml         # 环境配置文件
├── model.py                # 模型定义
├── data_loader.py          # 数据加载
├── strategy.py             # 训练策略
├── train.py                # 单策略训练脚本
├── train_comparison.py     # 多策略对比训练脚本
├── performance_test.py     # 性能测试脚本
├── results_comparison/     # 对比实验结果
└── performance_results/    # 性能测试结果
```

## 运行方法
### 单策略训练
```bash
cd experiments/regression
conda activate abr-regression
python train.py --dataset california --strategy abr
```

### 多策略对比训练
```bash
python train_comparison.py --dataset california --epochs 3 --batch_size 5
```

### 性能测试
```bash
python performance_test.py --dataset california --batch_size 5 --epochs 5 --num_runs 3
```

### 命令行参数
#### 通用参数
- `--dataset`: 选择数据集 (california/boston)，默认为california
- `--batch_size`: 批次大小，默认为5
- `--epochs`: 训练轮数，默认为3
- `--lr`: 学习率，默认为0.001
- `--hidden_dim`: 隐藏层维度，默认为64
- `--save_dir`: 结果保存目录

#### 策略特定参数
- `--loss_threshold`: ABR策略和滑动窗口策略的损失阈值，默认为0.5
- `--max_repeats`: 所有策略的最大重复次数，默认为5
- `--window_size`: 滑动窗口策略的窗口大小，默认为5

#### 性能测试特定参数
- `--num_runs`: 每种策略运行的次数，取平均值，默认为3

## TensorBoard集成
本项目已集成TensorBoard，用于可视化训练过程中的各项指标。

### 启动TensorBoard
训练完成后，在项目根目录(IterBatch)下使用以下命令启动TensorBoard：
```bash
tensorboard --logdir=experiments/regression/results_comparison/tensorboard_logs
```

如果在regression目录下运行命令，则使用：
```bash
tensorboard --logdir=results_comparison/tensorboard_logs
```

### 查看指标
在浏览器中访问输出的URL（通常为http://localhost:6006/），可以查看以下指标：
- 训练/测试损失
- 训练/测试RMSE
- 训练/测试MAE
- 各策略的重复训练次数

## 结果解读
### 输出文件
运行对比实验或性能测试后，将在指定的保存目录中生成以下文件：

1. 指标历史数据文件（`.npy`格式）
2. 可视化图表：
   - 损失曲线对比图 (loss_comparison.png)
   - RMSE曲线对比图 (rmse_comparison.png)
   - MAE曲线对比图 (mae_comparison.png)
   - 训练时间比较图 (training_time_comparison.png)
   - 重复训练次数比较图 (repeats_comparison.png)
   - 滑动窗口策略的损失趋势变化图 (window_trend.png)
3. 性能比较CSV文件 (performance_comparison.csv)

### 性能指标说明
1. **训练时间**：完成指定训练轮次所需的总时间（秒）
2. **平均重复次数/批次**：每个训练批次的平均重复训练次数
3. **测试RMSE**：在测试集上的均方根误差
4. **测试MAE**：在测试集上的平均绝对误差

## 示例结果分析
假设我们得到以下结果：

| 策略           | 平均训练时间 (秒) | 平均重复次数/批次 | 平均测试RMSE | 平均测试MAE |
|----------------|------------------|------------------|--------------|-------------|
| baseline       | 120.5            | 1.0              | 0.512        | 0.398       |
| abr            | 145.2            | 1.8              | 0.508        | 0.393       |
| learnable      | 180.7            | 2.2              | 0.510        | 0.395       |
| window         | 155.3            | 1.6              | 0.511        | 0.396       |

分析：
1. ABR策略在略微增加训练时间的情况下，取得了最低的RMSE和MAE，表明其在准确性上的优势
2. 可学习策略训练时间最长，但性能提升不明显，可能需要进一步调优
3. 滑动窗口策略平衡了训练时间和性能
4. 基线策略最快但准确性最低

## 注意事项
1. 首次运行可能需要下载数据集，请确保网络连接正常
2. 增加 `--num_runs` 参数可以提高结果的可靠性，但会增加总运行时间
3. 不同硬件环境下的绝对时间值可能有差异，建议关注相对比较结果
4. 如需调整实验参数，请修改命令行参数或脚本代码

## 扩展建议
1. 测试更多的超参数组合，如不同的学习率、批次大小
2. 增加其他性能指标，如训练过程中的内存使用
3. 对比不同模型架构下各策略的表现
4. 测试在更大规模数据集上的性能
5. 尝试结合多种策略的优点，开发新的混合策略