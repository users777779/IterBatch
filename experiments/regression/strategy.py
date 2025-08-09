"""训练策略实现

包含四种策略，均继承自 `BaseTrainingStrategy`：
- Baseline：每批次训练一次
- ABR：Adaptive Batch Repetition，根据初始 loss 自适应重复次数
- Learnable：可学习调度策略，在线用 ABR 的重复数作伪标签训练调度器
- SlidingWindow：根据最近窗口内损失趋势与波动度设定重复次数

公共能力：
- `evaluate`：在给定数据集上计算平均 Loss/RMSE/MAE
- `save_metrics*`：保存 epoch 级指标序列，便于落盘与绘图
"""

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

    def save_metrics_with_accuracy(self, epoch, train_loss, train_rmse, train_mae, train_accuracy, test_loss, test_rmse, test_mae, test_accuracy):
        """保存训练和测试指标（包含accuracy）

        Args:
            epoch: 当前 epoch 数
            train_loss: 训练损失
            train_rmse: 训练 RMSE
            train_mae: 训练 MAE
            train_accuracy: 训练 accuracy
            test_loss: 测试损失
            test_rmse: 测试 RMSE
            test_mae: 测试 MAE
            test_accuracy: 测试 accuracy
        """
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_rmse'].append(train_rmse)
        self.metrics_history['train_mae'].append(train_mae)
        self.metrics_history['train_accuracy'].append(train_accuracy)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_rmse'].append(test_rmse)
        self.metrics_history['test_mae'].append(test_mae)
        self.metrics_history['test_accuracy'].append(test_accuracy)


class BaselineStrategy(BaseTrainingStrategy):
    """基线策略：每个批次训练一次

    原理：
    - 每个 batch 仅做一次前向、一次反向、一次优化器更新
    - 作为其它策略的对照基线（计算量最小）
    返回：
    - (loss.item(), 1)
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
    def __init__(self, model, criterion, optimizer, loss_threshold=0.3, max_repeats=5):
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

        步骤：
        1) 初始前向得到 initial_loss
        2) 计算 repeats = clip( max(1, int(initial_loss / loss_threshold)), max_repeats )
        3) 重复执行该 batch 的前向+反向+更新 repeats 次
        4) 返回该 batch 内部训练的平均损失与重复次数
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
    预测最优的重复训练次数。此处提供一个可落地的在线监督方案：
    用 ABR 的重复次数作为伪标签，最小化回归损失，从而让调度器很快摆脱恒为 1 的输出。
    """
    def __init__(self, model, criterion, optimizer, scheduler_model, scheduler_optimizer, loss_threshold=0.3, max_repeats=5, epsilon=0.1, ema_gamma=0.8):
        """初始化可学习调度策略

        Args:
            model: 要训练的主模型
            criterion: 损失函数
            optimizer: 主模型的优化器
            scheduler_model: 用于预测重复次数的调度模型
            scheduler_optimizer: 调度模型的优化器
            loss_threshold: 用于生成伪标签（ABR）
            max_repeats: 伪标签的最大重复次数
        """
        super(LearnableSchedulingStrategy, self).__init__(model, criterion, optimizer)
        self.scheduler_model = scheduler_model  # 用于预测重复次数的模型
        self.scheduler_optimizer = scheduler_optimizer  # 调度模型的优化器
        self.scheduler_criterion = nn.MSELoss()  # 调度模型的损失函数
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats
        self.batch_history = []  # 记录每个批次的训练历史
        self.epsilon = float(epsilon)  # ε-greedy 探索率
        self.ema_gamma = float(ema_gamma)  # 输出 EMA 平滑系数
        self._ema_repeat = None  # EMA 的内部状态

        # 轻微偏置初始化：鼓励初期输出不为0，避免恒为1
        with torch.no_grad():
            if hasattr(self.scheduler_model, 'fc3') and hasattr(self.scheduler_model.fc3, 'bias'):
                self.scheduler_model.fc3.bias.fill_(2.0)

    def _compute_pseudo_label(self, initial_loss_value: float) -> int:
        """基于 ABR 规则生成伪标签重复次数"""
        if initial_loss_value > self.loss_threshold:
            return max(1, min(int(initial_loss_value / self.loss_threshold), self.max_repeats))
        return 1

    def train_batch(self, inputs, targets):
        """训练一个批次的数据（使用调度模型预测重复次数并在线监督训练）

        步骤：
        1) 计算 initial_loss 与 initial_accuracy≈1/(sqrt(MSE)+eps)
        2) 调度器前向输出 raw_repeat（实数）并用 ABR 规则生成的重复数作为伪标签做回归训练
        3) 对 raw_repeat 做 EMA 平滑得到 smoothed_repeat；以 ε 概率随机采样重复数（1..5），否则对平滑值四舍五入
        4) 重复训练该 batch repeat_count 次；记录历史（含 scheduler_loss/raw/ema/epsilon）
        返回：
        - (avg_loss, repeat_count)
        """
        self.model.train()  # 设置主模型为训练模式
        self.scheduler_model.train()  # 设置调度模型为训练模式

        # 初始前向传播，获取损失
        initial_outputs = self.model(inputs)
        initial_loss = self.criterion(initial_outputs, targets)

        # 初始 accuracy 近似（供调度器输入使用）
        with torch.no_grad():
            initial_rmse = torch.sqrt(initial_loss)
            epsilon = 1e-8
            initial_accuracy = 1.0 / (initial_rmse + epsilon)

        # 调度器前向（带梯度）
        scheduler_input = torch.tensor([[initial_loss.item(), initial_accuracy.item()]], dtype=torch.float32)
        scheduler_output = self.scheduler_model(scheduler_input)  # 形状 [1, 1]
        scheduler_output_scalar = scheduler_output.squeeze(0).squeeze(0)

        # 伪标签：用 ABR 规则生成的重复次数
        label_repeat = self._compute_pseudo_label(initial_loss.item())
        label_tensor = torch.tensor(float(label_repeat), dtype=torch.float32)

        # 训练调度器（回归到伪标签）
        self.scheduler_optimizer.zero_grad()
        scheduler_loss = self.scheduler_criterion(scheduler_output_scalar, label_tensor)
        scheduler_loss.backward()
        self.scheduler_optimizer.step()

        # 将调度输出映射到合法重复次数 [1, 5]
        raw_repeat = float(scheduler_output_scalar.detach().cpu().item())

        # EMA 平滑
        if self._ema_repeat is None:
            self._ema_repeat = raw_repeat
        else:
            self._ema_repeat = self.ema_gamma * self._ema_repeat + (1.0 - self.ema_gamma) * raw_repeat

        smoothed_repeat = self._ema_repeat

        # ε-greedy 探索：以 ε 概率随机选择 1..5 的一个值
        if np.random.rand() < self.epsilon:
            predicted_repeat = np.random.randint(1, 6)
        else:
            predicted_repeat = int(smoothed_repeat + 0.5)

        repeat_count = max(1, min(predicted_repeat, 5))

        # 重复训练主模型
        total_loss = 0.0
        batch_history = []
        for _ in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            batch_history.append(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / repeat_count

        # 记录历史
        self.batch_history.append({
            'initial_loss': initial_loss.item(),
            'initial_accuracy': initial_accuracy.item(),
            'pseudo_label_repeat': label_repeat,
            'predicted_repeats': repeat_count,
            'scheduler_loss': float(scheduler_loss.detach().cpu().item()),
            'loss_history': batch_history,
            'raw_repeat': float(raw_repeat),
            'ema_repeat': float(smoothed_repeat),
            'epsilon': float(self.epsilon),
        })

        return avg_loss, repeat_count


class SlidingWindowStrategy(BaseTrainingStrategy):
    """滑动窗口策略：根据最近批次的损失趋势与波动度调整重复训练次数

    原理（规范化版）：
    - 趋势：trend = (last - first) / (n - 1)
    - 标准差：std_dev = std(window)
    - 标准化：z_loss = (initial_loss - mean(window)) / std_dev；norm_trend = trend / std_dev
    - 风险评分：当 norm_trend 超过阈值、loss 高于阈值、波动超过阈值时累计风险
    - 重复次数：additional = round(risk) 裁剪到 [0, max_repeats]；repeat = 1 + additional，再裁剪到不超过 5
    """
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        window_size=5,
        loss_threshold=0.3,
        max_repeats=1,
        trend_threshold=0.01,
        vol_threshold=0.1,
        window_min_size=3,
    ):
        """初始化滑动窗口策略

        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
            window_size: 滑动窗口长度
            loss_threshold: 兼容旧参数（不强依赖），参与风险评分（高损失加分）
            max_repeats: 允许的“附加重复次数”上限（additional repeats）
            trend_threshold: 趋势阈值（标准化后）
            vol_threshold: 波动阈值（标准差阈值）
            window_min_size: 计算趋势/方差的最小窗口长度
        """
        super(SlidingWindowStrategy, self).__init__(model, criterion, optimizer)
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        self.window_min_size = window_min_size
        self.loss_window = deque(maxlen=window_size)
        self.batch_history = []

    def _compute_trend_std(self, series):
        """返回简单趋势斜率与标准差"""
        n = len(series)
        if n < 2:
            return 0.0, 0.0
        trend = (series[-1] - series[0]) / max(1, (n - 1))
        std = float(np.std(series))
        return float(trend), std

    def train_batch(self, inputs, targets):
        """训练一个批次的数据（根据损失趋势/波动与当前损失综合决定重复次数）"""
        self.model.train()
        total_loss = 0.0

        # 初始前向传播，获取当前批次初始损失
        initial_outputs = self.model(inputs)
        initial_loss = float(self.criterion(initial_outputs, targets).item())

        # 默认仅一次（无附加重复）
        additional_repeats = 0

        # 基于窗口计算趋势/波动，并规范化地映射为 additional_repeats
        if len(self.loss_window) >= self.window_min_size:
            recent_losses = list(self.loss_window)
            trend, std_dev = self._compute_trend_std(recent_losses)
            mean_loss = float(np.mean(recent_losses))
            safe_std = max(std_dev, 1e-8)
            # 标准化当前损失与趋势
            z_loss = (initial_loss - mean_loss) / safe_std
            norm_trend = trend / safe_std

            # 风险评分（可调的线性组合，保持简单、可解释）
            risk = 0.0
            if norm_trend > self.trend_threshold:
                risk += (norm_trend - self.trend_threshold)
            if initial_loss > self.loss_threshold:
                # 高于阈值时按 z 分数加权
                risk += max(0.0, 0.5 * z_loss)
            if std_dev > self.vol_threshold:
                risk += 0.5 * (std_dev - self.vol_threshold) / max(self.vol_threshold, 1e-8)

            additional_repeats = int(np.clip(np.round(risk), 0, self.max_repeats))

            # 训练历史记录增强：便于诊断
            diagnostics = {
                'trend': float(trend),
                'std_dev': float(std_dev),
                'mean_loss': float(mean_loss),
                'z_loss': float(z_loss),
                'norm_trend': float(norm_trend),
                'risk': float(risk),
            }
        else:
            diagnostics = {
                'trend': 0.0,
                'std_dev': 0.0,
                'mean_loss': float(initial_loss),
                'z_loss': 0.0,
                'norm_trend': 0.0,
                'risk': 0.0,
            }

        # 总重复次数：1 + 附加重复；并做全局硬上限 5
        repeat_count = 1 + additional_repeats
        repeat_count = min(repeat_count, 5)

        # 重复训练当前批次
        batch_history = []
        for _ in range(repeat_count):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.item())
            batch_history.append(loss_value)
            total_loss += loss_value

        # 更新窗口
        self.loss_window.append(initial_loss)

        avg_loss = total_loss / repeat_count

        # 记录本批次历史（包含诊断信息）
        record = {
            'initial_loss': float(initial_loss),
            'repeat_count': int(repeat_count),
            'loss_history': batch_history,
        }
        record.update(diagnostics)
        self.batch_history.append(record)

        return avg_loss, repeat_count