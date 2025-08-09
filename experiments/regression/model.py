"""模型定义

包含：
- `MLP`：用于回归任务的多层感知器（两层隐藏层 + Dropout）
- `SchedulerMLP`：用于 Learnable 策略的调度网络（输入为 [loss, accuracy]）

设计要点：
- 适度的 Dropout(0.2) 用于抑制过拟合
- 提供 `clone()` 方便复制结构参数（如需对比备份）
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知器模型，用于回归任务"""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def clone(self):
        """创建模型的深拷贝"""
        return MLP(self.fc1.in_features, self.fc1.out_features, self.fc3.out_features)


class SchedulerMLP(nn.Module):
    """调度模型，接收 [loss, r2] 作为输入"""
    def __init__(self, hidden_dim, output_dim=1):
        super(SchedulerMLP, self).__init__()
        # 输入维度为2 (loss, r2)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class WindowPolicyMLP(nn.Module):
    """滑动窗口策略的可学习决策网络

    输入特征建议： [z_loss, norm_trend, std_dev, mean_loss, initial_loss, delta_cur_mean]
    输出： 预测附加重复次数（实数，训练时会回归到整数目标）
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, output_dim: int = 1):
        super(WindowPolicyMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x