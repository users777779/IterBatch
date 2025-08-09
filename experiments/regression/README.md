# 回归任务实验（IterBatch/regression）

本目录实现了在加州房价数据集上的回归基准，用于对比不同的“批次重复训练”（Batch Repetition）策略，包括基线、ABR、自学习调度与滑动窗口策略，并通过 TensorBoard 统一记录指标。

## 目录结构

- `train.py`: 统一训练脚本（支持选择并对比多种策略）
- `strategy.py`: 策略实现（Baseline/ABR/Learnable/SlidingWindow）
- `model.py`: 主模型 `MLP` 与调度模型 `SchedulerMLP`
- `data_loader.py`: 数据加载与标准化（加州房价）
- `environment.yml`: Conda 环境定义

## 环境准备

```bash
conda env create -f environment.yml
conda activate abr-regression
# 如需 GPU，请根据本机 CUDA 版本安装匹配的 PyTorch
```

## 数据集与预处理

- 使用 `sklearn.datasets.fetch_california_housing` 获取 California Housing 数据集（当前仅实现该数据集）
- `StandardScaler` 标准化特征
- 随机划分训练/验证/测试集：`test_size` 默认 0.2、`val_ratio` 默认 0.0（>0 时启用验证集与早停）
- `DataLoader`：训练集 `shuffle=True`，验证/测试集 `shuffle=False`
- 提供 `get_train_loader() / get_val_loader() / get_test_loader()`

## 模型

- `MLP(input_dim, hidden_dim, output_dim=1)`：两层全连接 + ReLU + Dropout(0.2)
- `SchedulerMLP(hidden_dim, output_dim=1)`：输入为 2 维 `[loss, accuracy]`，用于预测批次重复次数

## 策略设计与实现

所有策略均继承自 `BaseTrainingStrategy`，对外暴露统一的 `train_batch(inputs, targets)` 接口，返回 `(avg_loss, repeat_count)`。

- 基线策略 Baseline
  - 每个批次只训练一次（一次前向 + 反向 + 更新）

- ABR 策略 Adaptive Batch Repetition
  - 先做一次前向计算初始损失 `initial_loss`
  - 重复次数计算：`repeat_count = max(1, min(int(initial_loss / loss_threshold), max_repeats))`
  - 对“难批次”（损失高）分配更多重复训练，易批次分配更少

- 可学习调度策略 Learnable Scheduling
  - 计算初始损失 `initial_loss` 与 `initial_accuracy≈1/(sqrt(MSE)+eps)` 作为调度器输入
  - 在线监督学习：用 ABR 规则生成重复数作为伪标签训练调度器（回归损失）
  - 输出稳定与探索：对调度原始输出做 EMA 平滑（`ema_gamma`），并加入 ε-greedy 探索（`epsilon`）
  - 将平滑后的输出四舍五入并裁剪到 [1,5] 作为重复次数；初始化对输出做轻微偏置避免恒为 1
  - 按预测重复次数训练主模型，并记录批次历史（包含 `scheduler_loss/raw_repeat/ema_repeat/epsilon` 等诊断信息）

- 滑动窗口策略 Sliding Window
  - 维护最近 `window_size` 个批次初始损失，计算简单斜率 `trend=(last-first)/(n-1)` 与窗口标准差 `std_dev`
  - 将当前损失做 z-score 标准化：`z_loss=(initial_loss-mean)/std`，并将 `trend/std` 作为 `norm_trend`
  - 风险评分：当 `norm_trend` 超阈值（上升趋势）、`initial_loss` 高于阈值（困难批次）、`std_dev` 超过阈值（波动高）时累加风险
  - `additional_repeats=round(risk)` 并裁剪到 `[0, max_repeats]`，总重复次数 `repeat_count=1+additional_repeats`，再裁剪到不超过 5
  - 批次历史记录包含诊断字段（`trend/std_dev/mean_loss/z_loss/norm_trend/risk`）

参数入口（来自 `train.py --help`）：

- `--strategies`: `baseline | abr | learnable | window`（可多选）
- `--dataset`: 数据集名。当前代码仅实现 `california`，传入 `boston` 会报错
- `--batch_size`: 批大小，默认 64
- `--epochs`: 训练轮数，默认 80
- `--lr`: 学习率，默认 1e-3（主模型与调度模型共用该值）
- `--hidden_dim`: 主模型隐藏维度，默认 64
- `--loss_threshold`: ABR/SlidingWindow 的损失阈值，默认 0.3
- `--max_repeats`: ABR/SlidingWindow 的最大重复次数上限（每批次），默认 5
- `--window_size`: 滑动窗口大小，默认 5
  
  Learnable 专用：
  - `--epsilon`: ε-greedy 探索率，默认 0.1（以 ε 概率随机选 1..5）
  - `--ema_gamma`: 调度输出的 EMA 平滑系数，默认 0.8（越大越平滑）
  - `--budget_ratio`: 额外重复次数预算比例（每 epoch 相对 baseline），当前为预留参数，后续版本将用于约束计算量
- `--val_ratio`: 验证集比例，默认 0.0（>0 时启用验证评估与早停）
- `--early_stopping`: 是否启用早停（基于验证集 Loss；无验证集时退化为测试集）
- `--patience`: 早停耐心值，默认 5
- `--save_dir`: 结果目录，默认 `results`

## 训练与对比

```bash
# 单策略
python train.py --strategies baseline --val_ratio 0.1 --early_stopping --patience 5

# 多策略对比（示例）
python train.py --strategies baseline abr learnable window --val_ratio 0.1 --early_stopping --patience 5
```

训练期间会输出每个 epoch 的训练/测试指标，并在非 Baseline 策略下额外打印“平均重复次数”（按批次平均）。

## 指标与记录

- 训练中按批次累计并在 epoch 级别统计：
  - Loss: MSE
  - RMSE: sqrt(MSE)
  - MAE: 平均绝对误差
  - Accuracy: 比例指标，定义为 `|pred - target| < 0.1`
- TensorBoard：位于 `results/tensorboard/<dataset>_<timestamp>`
  - 训练前会清空 `results/tensorboard/` 下旧的子目录，仅保留本次运行
  - 可视化：

    ```bash
    tensorboard --logdir=results
    # 浏览器打开 http://localhost:6006
    ```

- 逐批次写入（便于细粒度对比四策略）：
  - `.../Batch/Loss|RMSE|MAE|Accuracy`，非 Baseline 额外写入 `.../Batch/Repeats`
  - 每个 epoch 还会写入 `Train/Val/Test` 的聚合指标以及 `Repeats/Avg`

## 结果落盘

- 每种策略结束时会在 `save_dir` 下保存一个 `*_metrics.npy`，对应 `strategy.metrics_history`（训练循环已调用 `save_metrics_with_accuracy`，包含 Train/Val/Test 曲线与 Accuracy）

## 已知限制与注意事项

- `--dataset boston` 当前未在 `data_loader.py` 中实现，传入会报错；请使用 `--dataset california`
- TensorBoard 日志目录会在每次运行前被清空，仅保留本次实验日志
- `SlidingWindowStrategy` 的 `max_repeats` 为附加重复次数的上限，最终训练次数受裁剪（最多 5 次）影响

## 复现实验建议

在相同的随机划分与环境下，建议如下顺序执行并在 TensorBoard 对比：

1. 先跑 Baseline 作为对照：

```bash
python train.py --strategies baseline --val_ratio 0.1 --early_stopping --patience 5
```

1. 与 ABR/Window 对比：

```bash
python train.py --strategies baseline abr window --loss_threshold 0.3 --max_repeats 5 --window_size 5 --val_ratio 0.1 --early_stopping --patience 5
```

1. 加入可学习调度：

```bash
python train.py --strategies baseline learnable --val_ratio 0.1 --early_stopping --patience 5
```

配合 TensorBoard 对比 `Loss/RMSE/MAE/Accuracy` 与各策略的平均重复次数，观察收敛速度与最终误差差异。
