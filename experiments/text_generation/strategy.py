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
        # 训练进度信息（用于记录）
        self.current_epoch = 0
        self.current_step = 0
        # 异常样本记录（人工复核）
        self.outlier_records = []

    def _device(self):
        return next(self.model.parameters()).device

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def set_step(self, step: int):
        self.current_step = int(step)

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

    def _record_outlier(self, batch, init_val: float):
        try:
            # 仅采样第一个样本做简短预览，便于后续人工复核
            tokenizer = None
            if hasattr(self.model, 'get_tokenizer'):
                tokenizer = self.model.get_tokenizer()
            preview = None
            if tokenizer is not None and isinstance(batch.get('input_ids', None), torch.Tensor):
                ids = batch['input_ids'][0]
                if hasattr(ids, 'detach'):
                    ids = ids.detach().cpu()
                text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                preview = text[:200]
            self.outlier_records.append({
                'epoch': int(self.current_epoch),
                'step': int(self.current_step),
                'loss': float(init_val),
                'preview': preview,
            })
        except Exception:
            # 不因记录失败影响训练
            pass

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
            self._record_outlier(batch, init_val)
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
            self._record_outlier(batch, init_val)
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
            self._record_outlier(batch, init_val)
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
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        window_size: int = 5,
        loss_threshold: float = 2.0,
        max_repeats: int = 1,
        trend_threshold: float = 0.01,
        vol_threshold: float = 0.1,
        window_min_size: int = 3,
        volatility_mode: str = 'suppress',
        weight_trend: float = 1.0,
        weight_zloss: float = 0.5,
        weight_vol: float = 0.5,
        adaptive_window: bool = False,
        window_small: int | None = None,
        window_large: int | None = None,
        adapt_high_action: str = 'expand',
        adapt_low_action: str = 'shrink',
        vol_low_threshold: float | None = None,
    ):
        super().__init__(model, criterion, optimizer)
        self.window_size = int(window_size)
        self.loss_threshold = float(loss_threshold)
        self.max_repeats = int(max_repeats)
        self.trend_threshold = float(trend_threshold)
        self.vol_threshold = float(vol_threshold)
        self.window_min_size = int(window_min_size)
        self.loss_window = deque(maxlen=self.window_size)
        self.volatility_mode = volatility_mode
        self.weight_trend = float(weight_trend)
        self.weight_zloss = float(weight_zloss)
        self.weight_vol = float(weight_vol)
        self.adaptive_window = bool(adaptive_window)
        self.window_small = int(window_small) if window_small is not None else int(max(window_min_size, min(window_size, 3)))
        self.window_large = int(window_large) if window_large is not None else int(max(window_size, 2 * window_size))
        self.adapt_high_action = adapt_high_action
        self.adapt_low_action = adapt_low_action
        self.vol_low_threshold = float(vol_low_threshold) if vol_low_threshold is not None else float(0.5 * vol_threshold)

    def _compute_trend_std(self, series):
        n = len(series)
        if n < 2:
            return 0.0, 0.0
        trend = (series[-1] - series[0]) / max(1, (n - 1))
        std = float(np.std(series))
        return float(trend), std

    def train_batch(self, batch):
        self.model.train()
        with torch.no_grad():
            init_loss = self.compute_loss(batch)
            init_val = float(init_loss.item())

        if self._is_outlier(init_val):
            self.loss_history.append(init_val)
            self._record_outlier(batch, init_val)
            return init_val, 0

        additional = 0
        diagnostics = {}
        if len(self.loss_window) >= self.window_min_size:
            full_losses = list(self.loss_window)
            if self.adaptive_window:
                cur_std = float(np.std(full_losses))
                if cur_std > self.vol_threshold:
                    action = self.adapt_high_action
                elif cur_std < self.vol_low_threshold:
                    action = self.adapt_low_action
                else:
                    action = 'none'
                if action == 'expand':
                    effective_ws = min(len(full_losses), self.window_large)
                elif action == 'shrink':
                    effective_ws = min(len(full_losses), self.window_small)
                else:
                    effective_ws = min(len(full_losses), self.window_size)
            else:
                action = 'none'
                effective_ws = min(len(full_losses), self.window_size)

            recent = full_losses[-effective_ws:]
            trend, std_dev = self._compute_trend_std(recent)
            mean_loss = float(np.mean(recent))
            safe_std = max(std_dev, 1e-8)
            z_loss = (init_val - mean_loss) / safe_std
            norm_trend = trend / safe_std

            risk = 0.0
            if norm_trend > self.trend_threshold:
                risk += self.weight_trend * (norm_trend - self.trend_threshold)
            if init_val > self.loss_threshold:
                risk += max(0.0, self.weight_zloss * z_loss)
            if std_dev > self.vol_threshold:
                vol_term = self.weight_vol * (std_dev - self.vol_threshold) / max(self.vol_threshold, 1e-8)
                if self.volatility_mode == 'suppress':
                    risk -= vol_term
                else:
                    risk += vol_term
            additional = int(np.clip(np.round(risk), 0, self.max_repeats))
        
        repeats = min(1 + int(additional), 5)

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


class LearnableWindowStrategy(BaseTrainingStrategy):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        policy_model: nn.Module,
        policy_optimizer,
        window_size: int = 5,
        loss_threshold: float = 2.0,
        max_repeats: int = 1,
        trend_threshold: float = 0.01,
        vol_threshold: float = 0.1,
        window_min_size: int = 3,
        volatility_mode: str = 'suppress',
        policy_warmup_epochs: int = 0,
        policy_supervise_weight: float = 1.0,
        policy_main_weight: float = 1.0,
        policy_epsilon: float = 0.0,
        weight_trend: float = 1.0,
        weight_zloss: float = 0.5,
        weight_vol: float = 0.5,
        adaptive_window: bool = False,
        window_small: int | None = None,
        window_large: int | None = None,
        adapt_high_action: str = 'expand',
        adapt_low_action: str = 'shrink',
        vol_low_threshold: float | None = None,
    ):
        super().__init__(model, criterion, optimizer)
        assert policy_model is not None and policy_optimizer is not None
        # 滑窗参数
        self.window_size = int(window_size)
        self.loss_threshold = float(loss_threshold)
        self.max_repeats = int(max_repeats)
        self.trend_threshold = float(trend_threshold)
        self.vol_threshold = float(vol_threshold)
        self.window_min_size = int(window_min_size)
        self.loss_window = deque(maxlen=self.window_size)
        self.volatility_mode = volatility_mode
        self.weight_trend = float(weight_trend)
        self.weight_zloss = float(weight_zloss)
        self.weight_vol = float(weight_vol)
        self.adaptive_window = bool(adaptive_window)
        self.window_small = int(window_small) if window_small is not None else int(max(window_min_size, min(window_size, 3)))
        self.window_large = int(window_large) if window_large is not None else int(max(window_size, 2 * window_size))
        self.adapt_high_action = adapt_high_action
        self.adapt_low_action = adapt_low_action
        self.vol_low_threshold = float(vol_low_threshold) if vol_low_threshold is not None else float(0.5 * vol_threshold)

        # 可学习组件
        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizer
        self.policy_warmup_epochs = int(policy_warmup_epochs)
        self.policy_supervise_weight = float(policy_supervise_weight)
        self.policy_main_weight = float(policy_main_weight)
        self.policy_epsilon = float(policy_epsilon)
        self.batch_history = []

    def _compute_trend_std(self, series):
        n = len(series)
        if n < 2:
            return 0.0, 0.0
        trend = (series[-1] - series[0]) / max(1, (n - 1))
        std = float(np.std(series))
        return float(trend), std

    def _compute_window_diagnostics(self, initial_loss: float):
        if len(self.loss_window) >= self.window_min_size:
            full_losses = list(self.loss_window)
            if self.adaptive_window:
                cur_std = float(np.std(full_losses))
                if cur_std > self.vol_threshold:
                    action = self.adapt_high_action
                elif cur_std < self.vol_low_threshold:
                    action = self.adapt_low_action
                else:
                    action = 'none'
                if action == 'expand':
                    effective_ws = min(len(full_losses), self.window_large)
                elif action == 'shrink':
                    effective_ws = min(len(full_losses), self.window_small)
                else:
                    effective_ws = min(len(full_losses), self.window_size)
            else:
                action = 'none'
                effective_ws = min(len(full_losses), self.window_size)

            recent = full_losses[-effective_ws:]
            trend, std_dev = self._compute_trend_std(recent)
            mean_loss = float(np.mean(recent))
            safe_std = max(std_dev, 1e-8)
            z_loss = (initial_loss - mean_loss) / safe_std
            norm_trend = trend / safe_std
            risk = 0.0
            if norm_trend > self.trend_threshold:
                risk += self.weight_trend * (norm_trend - self.trend_threshold)
            if initial_loss > self.loss_threshold:
                risk += max(0.0, self.weight_zloss * z_loss)
            if std_dev > self.vol_threshold:
                vol_term = self.weight_vol * (std_dev - self.vol_threshold) / max(self.vol_threshold, 1e-8)
                if self.volatility_mode == 'suppress':
                    risk -= vol_term
                else:
                    risk += vol_term
            heuristic_additional = int(np.clip(np.round(risk), 0, self.max_repeats))
            diagnostics = {
                'trend': float(trend),
                'std_dev': float(std_dev),
                'mean_loss': float(mean_loss),
                'z_loss': float(z_loss),
                'norm_trend': float(norm_trend),
            }
        else:
            heuristic_additional = 0
            diagnostics = {
                'trend': 0.0,
                'std_dev': 0.0,
                'mean_loss': float(initial_loss),
                'z_loss': 0.0,
                'norm_trend': 0.0,
            }
        return heuristic_additional, diagnostics

    def train_batch(self, batch):
        self.model.train()
        self.policy_model.train()

        with torch.no_grad():
            init_loss = self.compute_loss(batch)
            init_val = float(init_loss.item())

        if self._is_outlier(init_val):
            self.loss_history.append(init_val)
            self._record_outlier(batch, init_val)
            return init_val, 0

        heuristic_additional, diagnostics = self._compute_window_diagnostics(init_val)

        # 策略网络输入：[z_loss, norm_trend, std_dev, mean_loss, initial_loss, delta_cur_mean]
        delta_cur_mean = init_val - diagnostics.get('mean_loss', init_val)
        feats = torch.tensor([[float(diagnostics.get('z_loss', 0.0)), float(diagnostics.get('norm_trend', 0.0)),
                               float(diagnostics.get('std_dev', 0.0)), float(diagnostics.get('mean_loss', init_val)),
                               float(init_val), float(delta_cur_mean)]], dtype=torch.float32, device=self._device())
        policy_out = self.policy_model(feats).squeeze(0).squeeze(0)
        raw_pred = float(policy_out.detach().cpu().item())
        if np.random.rand() < self.policy_epsilon:
            additional = np.random.randint(0, self.max_repeats + 1)
        else:
            additional = int(np.clip(round(raw_pred), 0, self.max_repeats))
        repeats = min(1 + additional, 5)

        total = 0.0
        batch_history = []
        for _ in range(repeats):
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            lv = float(loss.item())
            total += lv
            batch_history.append(lv)
        avg = total / max(1, repeats)

        self.loss_window.append(init_val)
        self.loss_history.append(init_val)

        # 主反馈 best_j
        with torch.no_grad():
            if len(batch_history) > 0:
                improvements = []
                for j in range(1, len(batch_history) + 1):
                    loss_j = batch_history[j - 1]
                    improvements.append(((init_val - loss_j) / j))
                best_j = int(np.argmax(improvements) + 1)
            else:
                best_j = 1

        # 策略训练（两阶段）
        self.policy_optimizer.zero_grad()
        policy_out2 = self.policy_model(feats).squeeze(0).squeeze(0)
        loss_terms = []
        if self.current_epoch < self.policy_warmup_epochs and self.policy_supervise_weight > 0:
            warmup_target = torch.tensor(float(heuristic_additional), dtype=torch.float32, device=self._device())
            loss_terms.append(self.policy_supervise_weight * nn.MSELoss()(policy_out2, warmup_target))
        else:
            main_target = torch.tensor(float(max(0, best_j - 1)), dtype=torch.float32, device=self._device())
            if self.policy_main_weight > 0:
                loss_terms.append(self.policy_main_weight * nn.MSELoss()(policy_out2, main_target))
            if self.policy_supervise_weight > 0:
                warmup_target = torch.tensor(float(heuristic_additional), dtype=torch.float32, device=self._device())
                loss_terms.append(self.policy_supervise_weight * nn.MSELoss()(policy_out2, warmup_target))
        policy_loss = torch.stack([lt if isinstance(lt, torch.Tensor) else torch.tensor(lt, device=self._device()) for lt in loss_terms]).sum() if len(loss_terms) else torch.tensor(0.0, device=self._device())
        policy_loss.backward()
        self.policy_optimizer.step()

        return avg, repeats
