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