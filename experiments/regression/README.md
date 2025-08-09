# 回归任务实验（IterBatch/regression）

本目录实现了在加州房价数据集上的回归基准，用于对比不同的“批次重复训练”（Batch Repetition）策略，包括基线、ABR、自学习调度与滑动窗口策略，并通过 TensorBoard 统一记录指标。

## 目录结构

- `train.py`: 统一训练脚本（支持选择并对比多种策略）
- `strategy.py`: 策略实现（Baseline/ABR/Learnable/SlidingWindow/LearnableWindow）
- `model.py`: 主模型 `MLP` 与调度模型 `SchedulerMLP`
- `presets.py`: 预设超参（快速复现最佳与备选配置）
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
- `SchedulerMLP(hidden_dim, output_dim=1)`：输入为 2 维 `[loss, r2]`，用于预测批次重复次数

## 策略设计与实现

所有策略均继承自 `BaseTrainingStrategy`，对外暴露统一的 `train_batch(inputs, targets)` 接口，返回 `(avg_loss, repeat_count)`。

- 基线策略 Baseline
  - 每个批次只训练一次（一次前向 + 反向 + 更新）

- ABR 策略 Adaptive Batch Repetition
  - 先做一次前向计算初始损失 `initial_loss`
  - 重复次数计算：`repeat_count = max(1, min(int(initial_loss / loss_threshold), max_repeats))`
  - 对“难批次”（损失高）分配更多重复训练，易批次分配更少

- 可学习ABR（Learnable Scheduling）
  - 计算初始损失 `initial_loss` 与 batch 近似 `R²` 作为调度器输入 `[loss, r2]`
  - 两阶段：
    - 预热（warmup）：用 ABR 规则生成的重复数作为伪标签监督调度器，避免早期过拟合主损失噪声
    - 主阶段：在每个 batch 内基于主模型损失序列选择“单位计算改进”最优的重复数 `best_j` 作为目标，对调度器进行反馈学习；可与 ABR 监督混合
  - 输出稳定与探索：对调度原始输出做 EMA 平滑（`ema_gamma`），并加入 ε-greedy 探索（`epsilon`）
  - 将平滑后的输出四舍五入并裁剪到 [1,5] 作为重复次数；初始化对输出做轻微偏置避免恒为 1
  - 批次历史包含 `scheduler_phase/best_j_from_feedback` 等诊断信息

- 滑动窗口策略 Sliding Window
  - 维护最近 `window_size` 个批次初始损失，计算简单斜率 `trend=(last-first)/(n-1)` 与窗口标准差 `std_dev`
  - 将当前损失做 z-score 标准化：`z_loss=(initial_loss-mean)/std`，并将 `trend/std` 作为 `norm_trend`
  - 波动处理模式：`--volatility_mode suppress|encourage` 分别表示当波动高时抑制或鼓励增加重复
  - 自适应窗口：`--adaptive_window` 启用后，基于当前波动大小在 `window_small|window_size|window_large` 中动态切换；
    - `--adapt_high_action expand|shrink`：高波动时扩大/缩小窗口；
    - `--adapt_low_action expand|shrink`：低波动时扩大/缩小窗口；
    - `--vol_low_threshold`：低波动阈值（默认 `0.5*vol_threshold`）。
  - 风险评分：当 `norm_trend` 超阈值（上升趋势）、`initial_loss` 高于阈值（困难批次）时增加风险；波动项按模式正负加权
  - `additional_repeats=round(risk)` 并裁剪到 `[0, max_repeats]`，总重复次数 `repeat_count=1+additional_repeats`，再裁剪到不超过 5
  - 可选“可学习决策网络”：`--use_window_policy` 启用后以小型 MLP 回归附加重复数，预热阶段用启发式 additional 监督，主阶段用主模型反馈的 `best_j-1` 监督，可与启发式混合
  - 批次历史记录包含诊断字段与 `used_policy/policy_phase/best_j_from_feedback` 等

参数入口（来自 `train.py --help`）：

- 通用：
  - `--strategies`: `baseline | abr | learnable | window | lwindow`（可多选）
  - `--dataset`: 数据集名（当前支持 `california`）
  - `--batch_size`: 默认 64；`--epochs`: 默认 80；`--lr`: 默认 1e-3；`--hidden_dim`: 默认 64
  - `--val_ratio`: 验证集比例；`--early_stopping` + `--patience`
  - `--save_dir`: 结果目录
  
- Learnable（可学习ABR）：
  - `--epsilon`, `--ema_gamma`, `--budget_ratio`
  - `--scheduler_warmup_epochs`, `--scheduler_supervise_weight`, `--scheduler_main_weight`

- Window：
  - `--loss_threshold`, `--max_repeats`, `--window_size`
  - `--volatility_mode`: suppress/encourage
  - 预设：`--preset window_best_v1 | window_stable_v1 | window_adaptive_v1`
  - 自适应：`--adaptive_window`, `--window_small`, `--window_large`, `--adapt_high_action`, `--adapt_low_action`, `--vol_low_threshold`
  - 权重与阈值：`--trend_threshold`, `--vol_threshold`, `--window_min_size`, `--weight_trend`, `--weight_zloss`, `--weight_vol`
  - 决策网络：`--use_window_policy` 或直接使用 `--strategies lwindow`；`--window_policy_hidden`, `--policy_*`

## 训练与对比

```bash
# 单策略
python train.py --strategies baseline --val_ratio 0.1 --early_stopping --patience 5

# 多策略对比（示例）
python train.py --strategies baseline abr learnable window --val_ratio 0.1 --early_stopping --patience 5
```

训练期间会输出每个 epoch 的训练/测试指标，并打印“平均重复次数”（所有策略均记录，Baseline 约为 1）。

## 指标与记录

- 训练中仅在 epoch 级别统计与记录（已停止逐 batch 写入，以减少噪声与日志体积）：
  - Loss: MSE
  - RMSE: sqrt(MSE)
  - MAE: 平均绝对误差
  - R2: 决定系数（按 batch 近似计算：`1 - SSE/SST`，再做样本加权求 epoch 均值）
- TensorBoard：位于 `results/tensorboard/<dataset>_<timestamp>/<strategy>`。
  - 每种策略使用独立子运行目录，并使用统一的标量标签（如 `Loss/Train`、`R2/Test`），便于在 TensorBoard 前端勾选多条曲线进行同图对比。
  - 可视化：
  
    ```bash
    tensorboard --logdir=results
    # 浏览器打开 http://localhost:6006
    ```

备注：当未设置验证集（`--val_ratio 0.0`）时，显示在 `Val` 名称下的曲线会等价于 `Test` 指标（训练脚本内部已退化为使用测试集做验证统计）。

- 逐批次写入（便于细粒度对比策略）：
  - `.../Batch/Loss|RMSE|MAE|R2`，非 Baseline 额外写入 `.../Batch/Repeats`
  - 每个 epoch 还会写入 `Train/Val/Test` 的聚合指标以及 `Repeats/Avg`

## 结果落盘

- 每种策略结束时会在 `save_dir` 下保存一个 `*_metrics.npy`，对应 `strategy.metrics_history`（包含 Train/Val/Test 曲线与 R2）

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

 1. 加入可学习 ABR：

```bash
python train.py --strategies baseline learnable --scheduler_warmup_epochs 5 --scheduler_supervise_weight 1.0 --scheduler_main_weight 1.0 --val_ratio 0.1 --early_stopping --patience 5
```

 1. 滑动窗口扩展/可学习：

```bash
python train.py --strategies window --volatility_mode suppress --window_size 5 --max_repeats 3 \
  --use_window_policy --window_policy_hidden 32 --policy_warmup_epochs 2 --policy_supervise_weight 1.0 --policy_main_weight 1.0 --policy_epsilon 0.1
```

 1. 独立可学习滑动窗口（lwindow）：

```bash
python train.py --strategies lwindow --volatility_mode suppress --window_size 5 --max_repeats 3 \
  --window_policy_hidden 32 --policy_warmup_epochs 2 --policy_supervise_weight 1.0 --policy_main_weight 1.0 --policy_epsilon 0.1
```

配合 TensorBoard 对比 `Loss/RMSE/MAE/R2` 与各策略的平均重复次数，观察收敛速度与最终误差差异。
