import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque


class BaseTrainingStrategy:
    """文本生成训练策略基类
    规范 train_batch 与 evaluate 接口；指标history用于保存loss/ppl等
    """
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_history = defaultdict(list)
        # outlier 阈值统计：基于历史损失的均值+K*std
        self.loss_history = []
        self.outlier_k = 2.5
        # 是否使用HF的device_map自动分配/CPU offload
        backbone = getattr(model, 'backbone', model)
        self.uses_device_map = hasattr(backbone, 'hf_device_map')

    def _device(self):
        return next(self.model.parameters()).device

    def _prepare_batch(self, batch):
        if self.uses_device_map:
            # 使用HF自动设备映射：不要手动搬运张量，保持CPU张量由transformers/accelerate分发
            return batch
        else:
            dev = self._device()
            return {
                'input_ids': batch['input_ids'].to(dev),
                'attention_mask': batch['attention_mask'].to(dev),
                'labels': batch['labels'].to(dev),
            }

    def compute_loss(self, batch):
        b = self._prepare_batch(batch)
        outputs = self.model(
            input_ids=b['input_ids'],
            attention_mask=b['attention_mask'],
            labels=b['labels'],
        )
        return outputs.loss

    def _is_outlier(self, initial_loss: float) -> bool:
        if len(self.loss_history) < 10:
            return False
        mu = float(np.mean(self.loss_history[-100:]))
        sd = float(np.std(self.loss_history[-100:]) + 1e-8)
        return initial_loss > (mu + self.outlier_k * sd)

    def train_batch(self, batch):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in data_loader:
            loss = self.compute_loss(batch)
            bs, seqlen = batch['input_ids'].shape
            total_loss += loss.item() * bs
            total_tokens += bs
        avg_loss = total_loss / max(1, total_tokens)
        ppl = float(np.exp(avg_loss))
        return avg_loss, ppl


class BaselineStrategy(BaseTrainingStrategy):
    def train_batch(self, batch):
        self.model.train()
        # 初始损失
        initial_loss = self.compute_loss(batch)
        init_val = float(initial_loss.item())

        # outlier 检测：不做反传，但记录
        if self._is_outlier(init_val):
            self.loss_history.append(init_val)
            return init_val, 0

        self.optimizer.zero_grad()
        initial_loss.backward()
        self.optimizer.step()

        self.loss_history.append(init_val)
        return init_val, 1


class ABRStrategy(BaseTrainingStrategy):
    def __init__(self, model, criterion, optimizer, loss_threshold=2.0, max_repeats=5):
        super().__init__(model, criterion, optimizer)
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats

    def train_batch(self, batch):
        self.model.train()
        with torch.no_grad():
            initial_loss = self.compute_loss(batch)
            init_val = float(initial_loss.item())

        if self._is_outlier(init_val):
            self.loss_history.append(init_val)
            return init_val, 0

        # 动态重复次数
        if init_val > self.loss_threshold:
            repeats = min(int(init_val / self.loss_threshold), self.max_repeats)
        else:
            repeats = 1

        total = 0.0
        for _ in range(repeats):
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
        avg = total / repeats
        self.loss_history.append(init_val)
        return avg, repeats


class LearnableSchedulingStrategy(BaseTrainingStrategy):
    def __init__(self, model, criterion, optimizer, scheduler_model, scheduler_optimizer):
        super().__init__(model, criterion, optimizer)
        self.scheduler_model = scheduler_model
        self.scheduler_optimizer = scheduler_optimizer
        self.scheduler_criterion = nn.MSELoss()

    def _sched_device(self):
        return next(self.scheduler_model.parameters()).device

    def train_batch(self, batch):
        self.model.train()
        # 初始指标
        with torch.no_grad():
            init_loss = self.compute_loss(batch)
            init_val = float(init_loss.item())
            init_ppl = float(np.exp(init_val))

        if self._is_outlier(init_val):
            self.loss_history.append(init_val)
            return init_val, 0

        # 预测重复次数 (1..5)
        with torch.no_grad():
            inp = torch.tensor([[init_val, init_ppl]], dtype=torch.float32, device=self._sched_device())
            pred = self.scheduler_model(inp).item()
            repeats = max(1, min(int(round(pred)), 5))

        # 主模型训练
        total = 0.0
        for _ in range(repeats):
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
        avg = total / repeats

        # 训练调度器
        self.scheduler_model.train()
        self.scheduler_optimizer.zero_grad()
        target = torch.tensor([[float(repeats)]], dtype=torch.float32, device=self._sched_device())
        inp = torch.tensor([[init_val, init_ppl]], dtype=torch.float32, device=self._sched_device())
        pred = self.scheduler_model(inp)
        sched_loss = self.scheduler_criterion(pred, target)
        sched_loss.backward()
        self.scheduler_optimizer.step()

        self.loss_history.append(init_val)
        return avg, repeats


class SlidingWindowStrategy(BaseTrainingStrategy):
    def __init__(self, model, criterion, optimizer, window_size=5, loss_threshold=2.0, max_repeats=1):
        super().__init__(model, criterion, optimizer)
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.max_repeats = max_repeats
        self.loss_window = deque(maxlen=window_size)

    def train_batch(self, batch):
        self.model.train()
        with torch.no_grad():
            init_loss = self.compute_loss(batch)
            init_val = float(init_loss.item())

        if self._is_outlier(init_val):
            self.loss_history.append(init_val)
            return init_val, 0

        repeats = 1
        if len(self.loss_window) >= self.window_size:
            recent = list(self.loss_window)
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            std_dev = float(np.std(recent))
            if trend > 0.01:
                if init_val > self.loss_threshold:
                    repeats = min(1 + self.max_repeats, 5)
                else:
                    repeats = min(2, 5)
            elif trend < -0.01:
                if init_val > self.loss_threshold:
                    repeats = min(2, 5)
                else:
                    repeats = 1
            else:
                if std_dev > 0.1:
                    if init_val > self.loss_threshold:
                        repeats = min(2 + self.max_repeats, 5)
                    else:
                        repeats = min(2, 5)
                else:
                    if init_val > self.loss_threshold:
                        repeats = min(1 + self.max_repeats, 5)
                    else:
                        repeats = 1

        total = 0.0
        for _ in range(repeats):
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
        avg = total / repeats

        self.loss_window.append(init_val)
        self.loss_history.append(init_val)
        return avg, repeats
