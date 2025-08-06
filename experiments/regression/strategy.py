import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入PyTorch优化器模块
import numpy as np  # 导入NumPy库用于数值计算
import time  # 导入时间模块用于计时
from collections import defaultdict, deque  # 导入集合模块中的defaultdict和deque

# 以下代码定义了四种不同的训练策略，用于回归任务的模型训练
# 所有策略都继承自BaseTrainingStrategy基类

class BaseTrainingStrategy:
    """训练策略基类

    所有训练策略的抽象基类，定义了训练、评估和指标保存的基本接口
    子类需要实现train_batch方法来定义具体的训练逻辑
    """
    def __init__(self, model, criterion, optimizer):
        """初始化训练策略

        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
        """
        self.model = model  # 要训练的模型
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器
        self.metrics_history = defaultdict(list)  # 用于存储训练和测试指标的历史记录
        self.train_time = 0  # 记录训练总时间

    def train_batch(self, inputs, targets):
        """训练一个批次的数据

        Args:
            inputs: 输入数据
            targets: 目标标签

        Returns:
            tuple: (平均损失值, 训练次数)

        Raises:
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError("子类必须实现train_batch方法")

    def evaluate(self, data_loader):
        """在测试集上评估模型

        Args:
            data_loader: 测试数据加载器

        Returns:
            tuple: (平均损失, 平均RMSE, 平均MAE)
        """
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0
        total_rmse = 0
        total_mae = 0
        count = 0

        with torch.no_grad():  # 评估时不计算梯度
            for inputs, targets in data_loader:
                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))  # 计算RMSE
                mae = torch.mean(torch.abs(outputs - targets))  # 计算MAE

                # 累加指标（考虑批次大小）
                total_loss += loss.item() * inputs.size(0)
                total_rmse += rmse.item() * inputs.size(0)
                total_mae += mae.item() * inputs.size(0)
                count += inputs.size(0)

        # 计算平均指标
        avg_loss = total_loss / count
        avg_rmse = total_rmse / count
        avg_mae = total_mae / count

        return avg_loss, avg_rmse, avg_mae

    def save_metrics(self, epoch, train_loss, train_rmse, train_mae, test_loss, test_rmse, test_mae):
        """保存训练和测试指标

        Args:
            epoch: 当前 epoch 数
            train_loss: 训练损失
            train_rmse: 训练 RMSE
            train_mae: 训练 MAE
            test_loss: 测试损失
            test_rmse: 测试 RMSE
            test_mae: 测试 MAE
        """
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_rmse'].append(train_rmse)
        self.metrics_history['train_mae'].append(train_mae)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_rmse'].append(test_rmse)
        self.metrics_history['test_mae'].append(test_mae)


class BaselineStrategy(BaseTrainingStrategy):
    """基线策略：每个批次只训练一次

    最简单的训练策略，每个批次数据只进行一次前向传播和反向传播
    用于与其他高级策略进行对比
    """
    def train_batch(self, inputs, targets):
        """训练一个批次的数据（只训练一次）

        Args:
            inputs: 输入数据
            targets: 目标标签

        Returns:
            tuple: (损失值, 训练次数=1)
        """
        self.model.train()  # 设置模型为训练模式
        self.optimizer.zero_grad()  # 清零梯度
        outputs = self.model(inputs)  # 前向传播
        loss = self.criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新权重
        return loss.item(), 1  # 返回损失值和训练次数


class ABRStrategy(BaseTrainingStrategy):
    """ABR策略：根据损失动态调整批次的重复训练次数

    Adaptive Batch Repetition (ABR)策略根据每个批次的初始损失值
    动态决定该批次的重复训练次数。损失值高的难训练批次会被训练更多次，
    损失值低的易训练批次则训练次数较少。
    """
    def __init__(self, model, criterion, optimizer, loss_threshold=0.5, max_repeats=5):
        """初始化ABR策略

        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
            loss_threshold: 损失阈值，用于判断批次难易程度
            max_repeats: 单个批次的最大重复训练次数
        """
        super(ABRStrategy, self).__init__(model, criterion, optimizer)
        self.loss_threshold = loss_threshold  # 损失阈值
        self.max_repeats = max_repeats  # 最大重复训练次数
        self.batch_history = []  # 记录每个批次的训练历史

    def train_batch(self, inputs, targets):
        """训练一个批次的数据（根据损失动态调整重复次数）

        Args:
            inputs: 输入数据
            targets: 目标标签

        Returns:
            tuple: (平均损失值, 重复训练次数)
        """
        self.model.train()  # 设置模型为训练模式
        total_loss = 0
        repeat_count = 0
        initial_outputs = self.model(inputs)  # 初始前向传播
        initial_loss = self.criterion(initial_outputs, targets).item()  # 计算初始损失

        # 根据初始损失决定重复次数
        if initial_loss > self.loss_threshold:
            # 损失越高，重复次数越多，但不超过最大重复次数
            repeat_count = min(int(initial_loss / self.loss_threshold), self.max_repeats)
        else:
            repeat_count = 1  # 损失低于阈值，只训练一次

        # 重复训练该批次
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()  # 清零梯度
            outputs = self.model(inputs)  # 前向传播
            loss = self.criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新权重
            batch_history.append(loss.item())  # 记录本次训练的损失
            total_loss += loss.item()  # 累加总损失

        avg_loss = total_loss / repeat_count  # 计算平均损失
        # 记录该批次的训练历史
        self.batch_history.append({
            'initial_loss': initial_loss,
            'repeat_count': repeat_count,
            'loss_history': batch_history
        })
        return avg_loss, repeat_count


class LearnableSchedulingStrategy(BaseTrainingStrategy):
    """可学习调度策略：使用神经网络学习最优训练次数

    该策略使用一个额外的调度模型来学习如何根据批次的初始损失值
    预测最优的重复训练次数。调度模型会根据训练过程中的反馈进行优化。
    """
    def __init__(self, model, criterion, optimizer, scheduler_model, scheduler_optimizer):
        """初始化可学习调度策略

        Args:
            model: 要训练的主模型
            criterion: 损失函数
            optimizer: 主模型的优化器
            scheduler_model: 用于预测重复次数的调度模型
            scheduler_optimizer: 调度模型的优化器
        """
        super(LearnableSchedulingStrategy, self).__init__(model, criterion, optimizer)
        self.scheduler_model = scheduler_model  # 用于预测重复次数的模型
        self.scheduler_optimizer = scheduler_optimizer  # 调度模型的优化器
        self.scheduler_criterion = nn.MSELoss()  # 调度模型的损失函数
        self.batch_history = []  # 记录每个批次的训练历史

    def train_batch(self, inputs, targets):
        """训练一个批次的数据（使用调度模型预测重复次数）

        Args:
            inputs: 输入数据
            targets: 目标标签

        Returns:
            tuple: (平均损失值, 预测的重复训练次数)
        """
        self.model.train()  # 设置主模型为训练模式
        self.scheduler_model.train()  # 设置调度模型为训练模式

        # 初始前向传播，获取损失
        initial_outputs = self.model(inputs)  # 主模型前向传播
        initial_loss = self.criterion(initial_outputs, targets)  # 计算初始损失

        # 使用调度模型预测重复次数
        with torch.no_grad():
            # 调度模型根据初始损失预测重复次数
            repeat_count = self.scheduler_model(initial_loss.unsqueeze(0)).item()
            # 限制重复次数在1-5次之间，并四舍五入
            repeat_count = max(1, min(int(repeat_count + 0.5), 5))

        # 重复训练
        total_loss = 0
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()  # 清零主模型梯度
            outputs = self.model(inputs)  # 主模型前向传播
            loss = self.criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新主模型权重
            batch_history.append(loss.item())  # 记录本次训练的损失
            total_loss += loss.item()  # 累加总损失

        avg_loss = total_loss / repeat_count  # 计算平均损失
        # 记录该批次的训练历史
        self.batch_history.append({
            'initial_loss': initial_loss.item(),
            'predicted_repeats': repeat_count,
            'loss_history': batch_history
        })

        return avg_loss, repeat_count


class SlidingWindowStrategy(BaseTrainingStrategy):
    """滑动窗口策略：根据最近批次的损失变化趋势调整重复训练次数

    该策略维护一个固定大小的滑动窗口，记录最近几个批次的损失值，
    并根据损失的变化趋势来调整当前批次的重复训练次数。
    如果损失呈上升趋势且当前损失高于阈值，则增加重复训练次数。
    """
    def __init__(self, model, criterion, optimizer, window_size=5, loss_threshold=0.5, max_repeats=1):
        """初始化滑动窗口策略

        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
            window_size: 滑动窗口的大小，即考虑的最近批次数量
            loss_threshold: 损失阈值，用于判断是否需要增加重复次数
            max_repeats: 单个批次的最大重复训练次数（不包括初始训练）
        """
        super(SlidingWindowStrategy, self).__init__(model, criterion, optimizer)
        self.window_size = window_size  # 滑动窗口大小
        self.loss_threshold = loss_threshold  # 损失阈值
        self.max_repeats = max_repeats  # 最大重复训练次数（不包括初始训练）
        self.loss_window = deque(maxlen=window_size)  # 维护最近window_size个批次的损失
        self.batch_history = []  # 记录每个批次的训练历史

    def train_batch(self, inputs, targets):
        """训练一个批次的数据（根据损失变化趋势调整重复次数）

        Args:
            inputs: 输入数据
            targets: 目标标签

        Returns:
            tuple: (平均损失值, 重复训练次数)
        """
        self.model.train()  # 设置模型为训练模式
        total_loss = 0
        repeat_count = 1  # 默认训练一次

        # 初始前向传播，获取损失
        initial_outputs = self.model(inputs)  # 前向传播
        initial_loss = self.criterion(initial_outputs, targets).item()  # 计算初始损失

        # 决定重复训练次数
        if len(self.loss_window) >= self.window_size:
            # 计算最近窗口内的损失变化趋势
            recent_losses = list(self.loss_window)
            # 使用线性回归计算趋势（斜率）
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

            # 如果损失呈上升趋势且当前损失高于阈值，则增加重复次数
            if trend > 0 and initial_loss > self.loss_threshold:
                # 最多重复max_repeats次（总共训练max_repeats+1次）
                repeat_count = min(1 + self.max_repeats, 5)  # 限制最大训练次数为5

        # 重复训练
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()  # 清零梯度
            outputs = self.model(inputs)  # 前向传播
            loss = self.criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新权重
            batch_history.append(loss.item())  # 记录本次训练的损失
            total_loss += loss.item()  # 累加总损失

        # 更新损失窗口
        self.loss_window.append(initial_loss)

        avg_loss = total_loss / repeat_count  # 计算平均损失
        # 计算当前窗口的趋势
        window_trend = 0
        if len(self.loss_window) >= 2:
            window_trend = np.polyfit(range(len(self.loss_window)), list(self.loss_window), 1)[0]

        # 记录该批次的训练历史
        self.batch_history.append({
            'initial_loss': initial_loss,
            'repeat_count': repeat_count,
            'loss_history': batch_history,
            'window_trend': window_trend
        })

        return avg_loss, repeat_count