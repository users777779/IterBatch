# IterBatch 图像分类实验

本实验实现了三种批训练策略的对比，探索自适应批重复（ABR）技术在CIFAR10图像分类中的效果。

## 实验策略

1. **Baseline**：标准训练，每个batch训练一次
2. **Loss-only ABR**：基于当前loss值决策是否重复训练
3. **Context ABR**：基于当前loss和历史loss均值决策

## 核心代码

### 决策网络
```python
class DecisionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 2)
        )
    
    def forward(self, x):
        return self.net(x)  # 返回logits
```

### ABR决策逻辑
```python
# 决策网络输出概率分布
action_logits = decider_network(decider_input)
action_probs = torch.softmax(action_logits, dim=-1)

# 根据概率分布随机采样决策
m = torch.distributions.Categorical(action_probs)
action = m.sample()  # 0: 不重复, 1: 重复

# 监督学习标签：使用采样的动作作为标签
label = action  # 决策网络学习自己做出的决策
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 20 | 训练轮数 |
| batch_size | 32 | 批大小 |
| lr | 0.001 | 主模型学习率 |
| scheduler_lr | 1e-4 | 决策网络学习率 |
| window_size | 10 | 历史loss窗口大小 |
| threshold | 无 | 使用采样动作作为标签 |

## 数据集和模型

- **数据集**：CIFAR10（10类，32×32×3）
- **主模型**：DeepCNN（3个卷积块 + 全连接层）
- **决策网络**：DecisionMLP（2层隐藏层）

## 运行方法

```bash
# 安装依赖
pip install torch torchvision numpy==1.26.4 matplotlib tensorboard

# 运行实验
cd experiments/image_classification
python main.py

# 查看结果
tensorboard --logdir=experiments/image_classification/result/runs
# 访问 http://localhost:6006
```

## 输出结果

- **TensorBoard**：6个独立曲线（每个实验的Accuracy和Loss）
- **PNG图表**：`result/iterbatch_exp_results.png`
- **数据路径**：`experiments/image_classification/data`

## 训练流程

### 监督学习ABR流程
1. **初始训练**：当前batch训练一次，得到loss
2. **决策网络**：loss输入决策网络，得到概率分布，采样得到动作
3. **决策网络学习**：使用采样的动作作为标签训练决策网络
4. **重复训练**：如果动作=1，重新训练当前batch（不更新决策网络）

### 实验特点

- ✅ 公平比较：相同初始权重
- ✅ 独立优化器：避免相互影响
- ✅ 自动清理：每次运行清理历史日志
- ✅ 详细记录：完整训练过程可视化
- ✅ 正确逻辑：决策网络学习自己的决策，重复训练不重复决策

## 故障排除

- **numpy版本冲突**：使用 `numpy==1.26.4`
- **内存不足**：减小 `batch_size`
- **训练缓慢**：使用GPU或减少epochs
- **TensorBoard无显示**：检查日志目录路径
