import torch
import torch.optim as optim
import numpy as np
import time
from collections import defaultdict

class BaseTrainingStrategy:
    """训练策略基类"""
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_history = defaultdict(list)
        self.train_time = 0

    def train_batch(self, inputs, targets):
        """训练一个批次的数据"""
        raise NotImplementedError("子类必须实现train_batch方法")

    def evaluate(self, data_loader):
        """在测试集上评估模型"""
        self.model.eval()
        total_loss = 0
        total_rmse = 0
        total_mae = 0
        count = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))
                mae = torch.mean(torch.abs(outputs - targets))

                total_loss += loss.item() * inputs.size(0)
                total_rmse += rmse.item() * inputs.size(0)
                total_mae += mae.item() * inputs.size(0)
                count += inputs.size(0)

        avg_loss = total_loss / count
        avg_rmse = total_rmse / count
        avg_mae = total_mae / count

        return avg_loss, avg_rmse, avg_mae

    def save_metrics(self, epoch, train_loss, train_rmse, train_mae, test_loss, test_rmse, test_mae):
        """保存训练和测试指标"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_rmse'].append(train_rmse)
        self.metrics_history['train_mae'].append(train_mae)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_rmse'].append(test_rmse)
        self.metrics_history['test_mae'].append(test_mae)


class BaselineStrategy(BaseTrainingStrategy):
    """基线策略：每个批次只训练一次"""
    def train_batch(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), 1  # 返回损失值和训练次数


class ABRStrategy(BaseTrainingStrategy):
    """ABR策略：根据损失动态调整批次的重复训练次数"""
    def __init__(self, model, criterion, optimizer, loss_threshold=0.5, max_repeats=5):
        super(ABRStrategy, self).__init__(model, criterion, optimizer)
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats
        self.batch_history = []  # 记录每个批次的训练历史

    def train_batch(self, inputs, targets):
        self.model.train()
        total_loss = 0
        repeat_count = 0
        initial_outputs = self.model(inputs)
        initial_loss = self.criterion(initial_outputs, targets).item()

        # 根据初始损失决定重复次数
        if initial_loss > self.loss_threshold:
            repeat_count = min(int(initial_loss / self.loss_threshold), self.max_repeats)
        else:
            repeat_count = 1

        # 重复训练
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            batch_history.append(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / repeat_count
        self.batch_history.append({
            'initial_loss': initial_loss,
            'repeat_count': repeat_count,
            'loss_history': batch_history
        })
        return avg_loss, repeat_count


class LearnableSchedulingStrategy(BaseTrainingStrategy):
    """可学习调度策略：使用神经网络学习最优训练次数"""
    def __init__(self, model, criterion, optimizer, scheduler_model, scheduler_optimizer):
        super(LearnableSchedulingStrategy, self).__init__(model, criterion, optimizer)
        self.scheduler_model = scheduler_model  # 用于预测重复次数的模型
        self.scheduler_optimizer = scheduler_optimizer
        self.scheduler_criterion = nn.MSELoss()
        self.batch_history = []

    def train_batch(self, inputs, targets):
        self.model.train()
        self.scheduler_model.train()

        # 初始前向传播，获取损失
        initial_outputs = self.model(inputs)
        initial_loss = self.criterion(initial_outputs, targets)

        # 使用调度模型预测重复次数
        with torch.no_grad():
            repeat_count = self.scheduler_model(initial_loss.unsqueeze(0)).item()
            repeat_count = max(1, min(int(repeat_count + 0.5), 5))  # 限制在1-5次之间

        # 重复训练
        total_loss = 0
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            batch_history.append(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / repeat_count
        self.batch_history.append({
            'initial_loss': initial_loss.item(),
            'predicted_repeats': repeat_count,
            'loss_history': batch_history
        })

        return avg_loss, repeat_count