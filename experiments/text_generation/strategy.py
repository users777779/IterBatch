import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
import time


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
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_history = defaultdict(list)
        self.train_time = 0

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
            tuple: (平均损失, 困惑度)
        """
        self.model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = input_ids.clone()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # 累加指标（考虑批次大小）
                total_loss += loss.item() * input_ids.size(0)
                count += input_ids.size(0)

        # 计算平均指标
        avg_loss = total_loss / count
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def save_metrics(self, epoch, train_loss, train_perplexity, test_loss, test_perplexity):
        """保存训练和测试指标

        Args:
            epoch: 当前 epoch 数
            train_loss: 训练损失
            train_perplexity: 训练困惑度
            test_loss: 测试损失
            test_perplexity: 测试困惑度
        """
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_perplexity'].append(train_perplexity)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_perplexity'].append(test_perplexity)


class BaselineStrategy(BaseTrainingStrategy):
    """基线策略：每个批次只训练一次

    最简单的训练策略，每个批次数据只进行一次前向传播和反向传播
    用于与其他高级策略进行对比
    """
    def train_batch(self, batch):
        """训练一个批次的数据（只训练一次）

        Args:
            batch: 包含input_ids和attention_mask的批次数据

        Returns:
            tuple: (损失值, 训练次数=1)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()  # 在语言模型中，目标是预测下一个词
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), 1


class ABRStrategy(BaseTrainingStrategy):
    """ABR策略：根据损失动态调整批次的重复训练次数

    Adaptive Batch Repetition (ABR)策略根据每个批次的初始损失值
    动态决定该批次的重复训练次数。损失值高的难训练批次会被训练更多次，
    损失值低的易训练批次则训练次数较少。
    """
    def __init__(self, model, criterion, optimizer, loss_threshold=2.0, max_repeats=5):
        """初始化ABR策略

        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
            loss_threshold: 损失阈值，用于判断批次难易程度
            max_repeats: 单个批次的最大重复训练次数
        """
        super(ABRStrategy, self).__init__(model, criterion, optimizer)
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats
        self.batch_history = []

    def train_batch(self, batch):
        """训练一个批次的数据（根据损失动态调整重复次数）

        Args:
            batch: 包含input_ids和attention_mask的批次数据

        Returns:
            tuple: (平均损失值, 重复训练次数)
        """
        self.model.train()
        total_loss = 0
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        
        # 初始前向传播
        with torch.no_grad():
            initial_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            initial_loss = initial_outputs.loss.item()

        # 根据初始损失决定重复次数
        if initial_loss > self.loss_threshold:
            # 损失越高，重复次数越多，但不超过最大重复次数
            repeat_count = min(int(initial_loss / self.loss_threshold), self.max_repeats)
        else:
            repeat_count = 1  # 损失低于阈值，只训练一次

        # 重复训练该批次
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            batch_history.append(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / repeat_count
        # 记录该批次的训练历史
        self.batch_history.append({
            'initial_loss': initial_loss,
            'repeat_count': repeat_count,
            'loss_history': batch_history
        })
        return avg_loss, repeat_count


class LearnableSchedulingStrategy(BaseTrainingStrategy):
    """可学习调度策略：使用神经网络学习最优训练次数

    该策略使用一个额外的调度模型来学习如何根据批次的初始损失值和困惑度
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
        self.scheduler_model = scheduler_model
        self.scheduler_optimizer = scheduler_optimizer
        self.scheduler_criterion = nn.MSELoss()
        self.batch_history = []

    def train_batch(self, batch):
        """训练一个批次的数据（使用调度模型预测重复次数）

        Args:
            batch: 包含input_ids和attention_mask的批次数据

        Returns:
            tuple: (平均损失值, 预测的重复训练次数)
        """
        self.model.train()
        self.scheduler_model.train()

        # 初始前向传播，获取损失和输出
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        
        initial_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        initial_loss = initial_outputs.loss
        
        # 计算初始困惑度
        with torch.no_grad():
            initial_perplexity = torch.exp(initial_loss)

        # 使用调度模型预测重复次数
        with torch.no_grad():
            # 调度模型根据初始损失和困惑度预测重复次数
            scheduler_input = torch.tensor([[initial_loss.item(), initial_perplexity.item()]], dtype=torch.float32)
            repeat_count = self.scheduler_model(scheduler_input).item()
            # 限制重复次数在1-5次之间，并四舍五入
            repeat_count = max(1, min(int(repeat_count + 0.5), 5))

        # 重复训练
        total_loss = 0
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            batch_history.append(loss.item())
            total_loss += loss.item()

        # 更新调度模型
        self.scheduler_optimizer.zero_grad()
        # 计算调度模型的损失（预测重复次数与实际重复次数的均方误差）
        scheduler_target = torch.tensor([[float(repeat_count)]], dtype=torch.float32)
        scheduler_input = torch.tensor([[initial_loss.item(), initial_perplexity.item()]], dtype=torch.float32)
        scheduler_output = self.scheduler_model(scheduler_input)
        scheduler_loss = self.scheduler_criterion(scheduler_output, scheduler_target)
        scheduler_loss.backward()
        self.scheduler_optimizer.step()

        avg_loss = total_loss / repeat_count
        # 记录该批次的训练历史
        self.batch_history.append({
            'initial_loss': initial_loss.item(),
            'initial_perplexity': initial_perplexity.item(),
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
    def __init__(self, model, criterion, optimizer, window_size=5, loss_threshold=2.0, max_repeats=1):
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
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats
        self.loss_window = deque(maxlen=window_size)
        self.batch_history = []

    def train_batch(self, batch):
        """训练一个批次的数据（根据损失变化趋势调整重复次数）

        Args:
            batch: 包含input_ids和attention_mask的批次数据

        Returns:
            tuple: (平均损失值, 重复训练次数)
        """
        self.model.train()
        total_loss = 0
        repeat_count = 1  # 默认训练一次

        # 初始前向传播，获取损失
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        
        initial_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        initial_loss = initial_outputs.loss.item()

        # 决定重复训练次数
        if len(self.loss_window) >= self.window_size:
            # 计算最近窗口内的损失变化趋势
            recent_losses = list(self.loss_window)
            # 使用线性回归计算趋势（斜率），确保有足够的数据点
            if len(recent_losses) >= 2:
                trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            else:
                trend = 0
            
            # 计算窗口内损失的标准差，衡量波动性
            std_dev = np.std(recent_losses)
            
            # 根据不同趋势调整重复次数
            # 考虑损失值、趋势和波动性来综合决定重复次数
            if trend > 0.01:  # 损失显著上升趋势
                if initial_loss > self.loss_threshold:
                    # 损失上升且当前损失高，增加重复次数
                    repeat_count = min(1 + self.max_repeats, 5)  # 限制最大训练次数为5
                else:
                    # 损失上升但当前损失不高，适度增加重复次数
                    repeat_count = min(2, 5)
            elif trend < -0.01:  # 损失显著下降趋势
                if initial_loss > self.loss_threshold:
                    # 损失下降但当前损失高，适度增加重复次数
                    repeat_count = min(2, 5)
                else:
                    # 损失下降且当前损失不高，减少重复次数
                    repeat_count = 1
            else:  # 趋势不明显
                if std_dev > 0.1:  # 波动较大
                    # 波动较大时，适度增加重复次数以稳定训练
                    if initial_loss > self.loss_threshold:
                        repeat_count = min(2 + self.max_repeats, 5)
                    else:
                        repeat_count = min(2, 5)
                else:
                    # 趋势不明显且波动小，根据损失值决定重复次数
                    if initial_loss > self.loss_threshold:
                        repeat_count = min(1 + self.max_repeats, 5)
                    else:
                        repeat_count = 1

        # 重复训练
        batch_history = []
        for i in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            batch_history.append(loss.item())
            total_loss += loss.item()

        # 更新损失窗口
        self.loss_window.append(initial_loss)

        avg_loss = total_loss / repeat_count
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