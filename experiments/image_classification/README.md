# IterBatch 图像分类实验

本实验实现了三种批训练策略的对比，探索**自适应批重复（Adaptive Batch Repetition, ABR）**技术在CIFAR10图像分类中的效果。

## 🎯 实验策略

1. **Baseline**：标准训练，每个batch训练一次
2. **SupLossNet (v1)**：基于loss特征的监督学习ABR，输入4维特征
3. **SupCtxNet (v2)**：基于context（历史信息）的监督学习ABR，输入6维特征

## 🏗️ 核心架构

### 主分类模型
```python
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3个卷积块：3→32→64→128通道
        # 每个块：Conv2d + BatchNorm + ReLU + MaxPool
        # 全连接层：128×4×4 → 512 → 256 → 10
        # Dropout防过拟合
```

### 决策网络
```python
class DecisionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # 输出：不重复(0) / 重复(1)
        )
```

### 特征工程
- **SupLossNet (v1)**: `[loss, acc, loss_ratio, acc_gap]` (4维)
- **SupCtxNet (v2)**: `[loss, avg_loss, acc, avg_acc, loss_ratio, acc_gap]` (6维)

## 🔄 ABR决策逻辑

### 监督信号生成
```python
def generate_supervision_signal(current_loss, current_acc, recent_losses, recent_accs, device):
    # 动态阈值策略
    loss_threshold_1 = 1.02  # 灾难性表现阈值
    loss_threshold_2 = 1.01  # 表现不佳阈值
    acc_threshold_1 = 0.995  # 准确率阈值1
    acc_threshold_2 = 0.998  # 准确率阈值2
    
    # 规则A: 灾难性表现
    if current_loss > avg_loss * loss_threshold_1 or current_acc < avg_acc * acc_threshold_1:
        should_repeat = True
    
    # 规则B: 表现不佳
    elif current_loss > avg_loss * loss_threshold_2 and current_acc < avg_acc * acc_threshold_2:
        should_repeat = True
    
    # 规则C: 退步惩罚
    elif current_loss > avg_loss and current_acc < avg_acc:
        should_repeat = True
```

### 智能重复训练
```python
if action == 1:  # 决策重复训练
    # 记录训练前状态
    loss_before = evaluate_batch()
    
    # 递增学习率重复训练（最多3次）
    for repeat_step in range(3):
        lr_multiplier = 1.2 + repeat_step * 0.2  # 1.2x, 1.4x, 1.6x
        train_with_increased_lr()
        
        # 检查是否改善
        if improvement > threshold:
            break  # 提前停止
```

## ⚙️ 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 50 | 训练轮数 |
| batch_size | 32 | 批大小 |
| lr | 0.001 | 主模型学习率 |
| scheduler_lr | 1e-4 | 决策网络学习率 |
| window_size | 10 | 历史表现窗口大小 |
| max_repeats | 50% | 每个epoch最大重复比例 |
| max_repeat_steps | 3 | 每次最多重复训练次数 |
| lr_multiplier | [1.2, 1.4, 1.6] | 递增学习率倍数 |

## 📊 数据集和模型

- **数据集**：CIFAR10（10类，32×32×3彩色图像）
- **主模型**：DeepCNN（3个卷积块 + 全连接层 + Dropout）
- **决策网络**：DecisionMLP（4层隐藏层 + LayerNorm + Dropout）

## 🚀 运行方法

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib tensorboard

# 运行实验
cd experiments/image_classification
python main.py

# 查看TensorBoard结果
tensorboard --logdir=result/runs
# 访问 http://localhost:6006
```

## 📈 输出结果

### TensorBoard可视化
- **Accuracy/Test**: 三个实验的准确率在同一图表中比较
- **Loss/Test**: 三个实验的损失在同一图表中比较
- **DecisionNet**: 决策网络效果指标
- **RepeatTraining**: 重复训练统计信息
- **Improvement**: 相对于Baseline的性能提升

### 静态图表
- `accuracy_comparison.png`: 三个实验的准确率对比
- `loss_comparison.png`: 三个实验的损失对比
- `iterbatch_exp_results.png`: 综合对比图

### 数据路径
- **模型输出**: `experiments/image_classification/result/`
- **TensorBoard日志**: `experiments/image_classification/result/runs/iterbatch_exp/`

## 🔬 训练流程

### 完整ABR流程
1. **初始训练**: 当前batch训练一次，得到loss和accuracy
2. **特征计算**: 计算loss_ratio、acc_gap等特征
3. **决策网络**: 输入特征到决策网络，得到重复训练概率
4. **监督学习**: 使用动态生成的监督信号训练决策网络
5. **智能重复**: 如果决策=1，使用递增学习率重复训练（最多3次）
6. **效果验证**: 检查是否改善，改善即停止
7. **历史更新**: 更新滑动窗口中的历史表现

### 实验特点

- ✅ **公平比较**: 三个实验使用相同的初始权重
- ✅ **独立优化器**: 避免模型间相互影响
- ✅ **智能决策**: 基于历史表现的动态阈值策略
- ✅ **递增学习率**: 重复训练时使用1.2x→1.4x→1.6x学习率
- ✅ **早期停止**: 达到改善阈值即停止重复训练
- ✅ **历史持久化**: 滑动窗口不重置，保持训练稳定性
- ✅ **详细监控**: 完整的训练过程可视化和统计分析

## 🧠 核心创新点

### 1. 动态阈值策略
- 基于历史表现自动调整触发阈值
- 避免固定阈值的局限性

### 2. 递增学习率重复训练
- 不是简单重复，而是用越来越强的学习率
- 智能停止机制避免过度训练

### 3. 双重监督学习
- 决策网络通过监督学习训练
- 监督信号基于历史表现动态生成

### 4. 智能资源管理
- 限制最大重复比例，避免资源浪费
- 早期停止机制提高训练效率

## 🔧 故障排除

- **CUDA内存不足**: 减小`batch_size`或使用CPU训练
- **训练缓慢**: 使用GPU或减少`epochs`
- **TensorBoard无显示**: 检查日志目录路径和端口占用
- **依赖冲突**: 使用虚拟环境隔离依赖

## 📚 相关论文

本项目实现了自适应批重复（ABR）技术的实验验证，相关技术可用于：
- 困难样本识别和重点训练
- 训练效率优化
- 自适应学习策略
- 计算资源智能分配

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

*最后更新: 2025-08-12*
