"""统一训练脚本

功能：
- 解析 CLI 参数，构建数据、模型与训练策略
- 训练循环：逐批训练；TensorBoard 仅记录 epoch 级指标
- TensorBoard：为每种策略使用独立子运行目录，并用统一标签（如 'Loss/Train'）便于同图对比
- 早停：基于 Val Loss（无验证集时退化为 Test Loss）
- 结果落盘：各策略 `metrics_history` 保存为 `*.npy`
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import time
from model import MLP
from data_loader import HousingDataLoader
from strategy import BaselineStrategy, ABRStrategy, LearnableSchedulingStrategy, SlidingWindowStrategy, LearnableWindowStrategy
from presets import apply_preset
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='回归任务统一实验')
    parser.add_argument('--dataset', type=str, default='california', choices=['california', 'boston'],
                        help='数据集选择 (california/boston)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'],
                        help='模型选择 (当前仅支持mlp)')
    parser.add_argument('--strategies', type=str, nargs='+', default=['baseline'], 
                         choices=['baseline', 'abr', 'learnable', 'window', 'lwindow'],
                         help='训练策略选择 (baseline/abr/learnable/window/lwindow)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--loss_threshold', type=float, default=0.3, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=5, help='ABR策略的最大重复次数')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口策略的窗口大小')
    parser.add_argument('--volatility_mode', type=str, default='suppress', choices=['suppress', 'encourage'], help='滑动窗口中对高波动的处理方式：抑制或鼓励重复')
    parser.add_argument('--trend_threshold', type=float, default=0.01, help='滑动窗口：趋势阈值（标准化后）')
    parser.add_argument('--vol_threshold', type=float, default=0.1, help='滑动窗口：波动阈值（标准差阈值）')
    parser.add_argument('--window_min_size', type=int, default=3, help='滑动窗口：计算趋势/方差的最小窗口长度')
    parser.add_argument('--weight_trend', type=float, default=1.0, help='滑动窗口：风险评分中趋势项权重')
    parser.add_argument('--weight_zloss', type=float, default=0.5, help='滑动窗口：风险评分中 z-loss 项权重')
    parser.add_argument('--weight_vol', type=float, default=0.5, help='滑动窗口：风险评分中波动项权重')
    # 自适应窗口
    parser.add_argument('--adaptive_window', action='store_true', help='是否启用基于波动的自适应窗口长度选择')
    parser.add_argument('--window_small', type=int, default=None, help='自适应窗口的较小窗口长度')
    parser.add_argument('--window_large', type=int, default=None, help='自适应窗口的较大窗口长度')
    parser.add_argument('--adapt_high_action', type=str, default='expand', choices=['expand','shrink'], help='波动高于阈值时：扩大或缩小窗口')
    parser.add_argument('--adapt_low_action', type=str, default='shrink', choices=['expand','shrink'], help='波动低于较低阈值时：扩大或缩小窗口')
    parser.add_argument('--vol_low_threshold', type=float, default=None, help='定义低波动阈值（默认 0.5*vol_threshold）')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='验证集比例（0~0.5），用于早停与超参选择')
    parser.add_argument('--early_stopping', action='store_true', help='是否启用早停（基于验证集或测试集）')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值（若指标无提升的最大连续 epoch 数）')
    # Learnable 策略增强参数
    parser.add_argument('--epsilon', type=float, default=0.1, help='Learnable 策略的探索率 ε（ε-greedy）')
    parser.add_argument('--ema_gamma', type=float, default=0.8, help='Learnable 输出重复次数的 EMA 平滑系数 γ')
    parser.add_argument('--budget_ratio', type=float, default=None, help='Learnable 额外重复次数预算比例（相对 baseline，每 epoch）例如 0.5/1/2；None 表示不限制')
    parser.add_argument('--scheduler_warmup_epochs', type=int, default=5, help='Learnable 调度器的 ABR 伪标签预热轮数')
    parser.add_argument('--scheduler_supervise_weight', type=float, default=1.0, help='Learnable 中 ABR 监督项的损失权重')
    parser.add_argument('--scheduler_main_weight', type=float, default=1.0, help='Learnable 中主模型反馈项的损失权重')

    # Window 可学习策略参数
    parser.add_argument('--use_window_policy', action='store_true', help='是否启用滑动窗口的可学习决策网络')
    parser.add_argument('--window_policy_hidden', type=int, default=32, help='滑动窗口策略决策网络的隐藏维度')
    parser.add_argument('--policy_warmup_epochs', type=int, default=0, help='窗口策略网络预热轮数（用启发式 additional 监督）')
    parser.add_argument('--policy_supervise_weight', type=float, default=1.0, help='窗口策略启发式监督损失权重')
    parser.add_argument('--policy_main_weight', type=float, default=1.0, help='窗口策略主反馈损失权重')
    parser.add_argument('--policy_epsilon', type=float, default=0.0, help='窗口策略 ε-greedy 探索率')
    # 预设
    parser.add_argument('--preset', type=str, default=None, help='使用命名预设（见 presets.py）')
    return parser.parse_args()

def get_model(input_dim, hidden_dim, model_type='mlp'):
    """创建模型实例"""
    if model_type == 'mlp':
        return MLP(input_dim, hidden_dim)
    else:
        raise ValueError(f"不支持的模型: {model_type}")

def get_strategy(model, criterion, optimizer, strategy_name, args):
    """根据策略名称创建对应的策略实例"""
    if strategy_name == 'baseline':
        return BaselineStrategy(model, criterion, optimizer)
    elif strategy_name == 'abr':
        return ABRStrategy(model, criterion, optimizer, args.loss_threshold, args.max_repeats)
    elif strategy_name == 'learnable':
        # 创建调度模型（接收loss和accuracy作为输入）
        from model import SchedulerMLP
        scheduler_model = SchedulerMLP(32, 1)
        scheduler_optimizer = optim.Adam(scheduler_model.parameters(), lr=args.lr)
        return LearnableSchedulingStrategy(
            model,
            criterion,
            optimizer,
            scheduler_model,
            scheduler_optimizer,
            loss_threshold=args.loss_threshold,
            max_repeats=args.max_repeats,
            epsilon=args.epsilon,
            ema_gamma=args.ema_gamma,
            scheduler_warmup_epochs=args.scheduler_warmup_epochs,
            scheduler_supervise_weight=args.scheduler_supervise_weight,
            scheduler_main_weight=args.scheduler_main_weight,
        )
    elif strategy_name == 'window':
        # 可选可学习决策网络
        policy_model = None
        policy_optimizer = None
        if args.use_window_policy:
            from model import WindowPolicyMLP
            policy_model = WindowPolicyMLP(hidden_dim=args.window_policy_hidden)
            policy_optimizer = optim.Adam(policy_model.parameters(), lr=args.lr)
        return SlidingWindowStrategy(
            model,
            criterion,
            optimizer,
            args.window_size,
            args.loss_threshold,
            args.max_repeats,
            trend_threshold=args.trend_threshold,
            vol_threshold=args.vol_threshold,
            window_min_size=args.window_min_size,
            volatility_mode=args.volatility_mode,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            policy_warmup_epochs=args.policy_warmup_epochs,
            policy_supervise_weight=args.policy_supervise_weight,
            policy_main_weight=args.policy_main_weight,
            policy_epsilon=args.policy_epsilon,
            weight_trend=args.weight_trend,
            weight_zloss=args.weight_zloss,
            weight_vol=args.weight_vol,
            adaptive_window=args.adaptive_window,
            window_small=args.window_small,
            window_large=args.window_large,
            adapt_high_action=args.adapt_high_action,
            adapt_low_action=args.adapt_low_action,
            vol_low_threshold=args.vol_low_threshold,
        )
    elif strategy_name == 'lwindow':
        # 独立的可学习滑动窗口策略（强制需要决策网络）
        from model import WindowPolicyMLP
        policy_model = WindowPolicyMLP(hidden_dim=args.window_policy_hidden)
        policy_optimizer = optim.Adam(policy_model.parameters(), lr=args.lr)
        return LearnableWindowStrategy(
            model,
            criterion,
            optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            window_size=args.window_size,
            loss_threshold=args.loss_threshold,
            max_repeats=args.max_repeats,
            trend_threshold=args.trend_threshold,
            vol_threshold=args.vol_threshold,
            window_min_size=args.window_min_size,
            volatility_mode=args.volatility_mode,
            policy_warmup_epochs=args.policy_warmup_epochs,
            policy_supervise_weight=args.policy_supervise_weight,
            policy_main_weight=args.policy_main_weight,
            policy_epsilon=args.policy_epsilon,
            weight_trend=args.weight_trend,
            weight_zloss=args.weight_zloss,
            weight_vol=args.weight_vol,
            adaptive_window=args.adaptive_window,
            window_small=args.window_small,
            window_large=args.window_large,
            adapt_high_action=args.adapt_high_action,
            adapt_low_action=args.adapt_low_action,
            vol_low_threshold=args.vol_low_threshold,
        )
    else:
        raise ValueError(f"不支持的策略: {strategy_name}")

def calculate_metrics(outputs, targets):
    """计算评估指标"""
    loss = torch.mean((outputs - targets) ** 2)
    rmse = torch.sqrt(loss)
    mae = torch.mean(torch.abs(outputs - targets))
    # R^2: 1 - SSE/SST（按当前 batch 计算，作近似）
    sse = torch.sum((outputs - targets) ** 2)
    mean_target = torch.mean(targets)
    sst = torch.sum((targets - mean_target) ** 2) + 1e-8
    r2 = 1.0 - sse / sst
    return loss.item(), rmse.item(), mae.item(), r2.item()


def evaluate_epoch(model, criterion, data_loader):
    """在给定数据集上评估 epoch 指标"""
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss, rmse, mae, _ = calculate_metrics(outputs, targets)
            total_loss += loss * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_mae += mae * inputs.size(0)
            count += inputs.size(0)
    return total_loss / count, total_rmse / count, total_mae / count

def train_single_strategy(strategy_name, args, data_loader, input_dim, writer):
    """训练单个策略"""
    print(f"开始训练 {strategy_name} 策略...")
    
    # 创建模型
    model = get_model(input_dim, args.hidden_dim, args.model)
    
    # 创建优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建训练策略
    strategy = get_strategy(model, criterion, optimizer, strategy_name, args)
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
        # 训练循环
    start_time = time.time()
    
    best_val_metric = float('inf')
    epochs_no_improve = 0

    # Learnable 预算初始化（按 baseline 估计：每批 1 次 backward）
    if strategy_name == 'learnable' and args.budget_ratio is not None:
        estimated_baseline_backprops = len(data_loader.get_train_loader()) * 1
        epoch_budget = int(max(0, args.budget_ratio * estimated_baseline_backprops))
        budget_remaining = epoch_budget
    else:
        epoch_budget = None
        budget_remaining = None

    for epoch in range(args.epochs):
        # 将 epoch 信息通知策略（用于两阶段训练/预热）
        if hasattr(strategy, 'set_epoch'):
            strategy.set_epoch(epoch)
        epoch_start_time = time.time()
        total_loss = 0
        total_rmse = 0
        total_mae = 0
        total_accuracy = 0
        total_repeats = 0
        count = 0
        
        # 训练一个epoch（仅记录 epoch 级指标，不再进行逐 batch 的 TensorBoard 记录）
        for step_idx, (inputs, targets) in enumerate(train_loader):
            # 训练一个批次
            # 若设置了预算，且 Learnable 策略，传入预算剩余信息（可选接口扩展）
            loss, repeats = strategy.train_batch(inputs, targets)
            outputs = model(inputs)
            batch_loss, batch_rmse, batch_mae, batch_accuracy = calculate_metrics(outputs, targets)
            
            total_loss += batch_loss * inputs.size(0)
            total_rmse += batch_rmse * inputs.size(0)
            total_mae += batch_mae * inputs.size(0)
            total_accuracy += batch_accuracy * inputs.size(0)
            total_repeats += repeats
            count += inputs.size(0)
        
        avg_loss = total_loss / count
        avg_rmse = total_rmse / count
        avg_mae = total_mae / count
        avg_accuracy = total_accuracy / count
        avg_repeats = total_repeats / len(train_loader)
        
        # 在验证/测试集上评估
        if val_loader is not None:
            val_loss, val_rmse, val_mae = strategy.evaluate(val_loader)
        else:
            val_loss, val_rmse, val_mae = strategy.evaluate(test_loader)

        test_loss, test_rmse, test_mae = strategy.evaluate(test_loader)
        
        # 计算测试集 R^2
        total_test_accuracy = 0
        test_count = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, _, _, test_accuracy = calculate_metrics(outputs, targets)
                total_test_accuracy += test_accuracy * inputs.size(0)
                test_count += inputs.size(0)
        test_accuracy = total_test_accuracy / test_count

        # 计算验证集 R^2（若有）
        val_accuracy = None
        if val_loader is not None:
            total_val_accuracy = 0
            val_count = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    _, _, _, _acc = calculate_metrics(outputs, targets)
                    total_val_accuracy += _acc * inputs.size(0)
                    val_count += inputs.size(0)
            val_accuracy = total_val_accuracy / val_count
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 记录 TensorBoard：统一标签，便于将多策略绘制在同一图中
        # 在第一个 epoch 记录实验配置
        if epoch == 0:
            writer.add_text('Experiment/Config', f"Strategy: {strategy_name}, Dataset: {args.dataset}, Batch Size: {args.batch_size}, Learning Rate: {args.lr}")
        
        # Loss metrics
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        
        # RMSE metrics
        writer.add_scalar('RMSE/Train', avg_rmse, epoch)
        writer.add_scalar('RMSE/Val', val_rmse, epoch)
        writer.add_scalar('RMSE/Test', test_rmse, epoch)
        
        # MAE metrics
        writer.add_scalar('MAE/Train', avg_mae, epoch)
        writer.add_scalar('MAE/Val', val_mae, epoch)
        writer.add_scalar('MAE/Test', test_mae, epoch)
        
        # R2 metrics
        writer.add_scalar('R2/Train', avg_accuracy, epoch)
        if val_accuracy is not None:
            writer.add_scalar('R2/Val', val_accuracy, epoch)
        writer.add_scalar('R2/Test', test_accuracy, epoch)
        
        # 训练时间
        writer.add_scalar('Time/Epoch', epoch_time, epoch)
        
        # 平均重复次数（对 baseline 也记录，值约为 1，便于同图对比）
        writer.add_scalar('Repeats/Avg', avg_repeats, epoch)
            
        print(f"{strategy_name} - Epoch {epoch+1}/{args.epochs}, 训练时间: {epoch_time:.2f}秒")
        print(f"  训练损失: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_accuracy:.4f}")
        if val_loader is not None:
            print(f"  验证损失: {val_loss:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_accuracy:.4f}")
        print(f"  测试损失: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_accuracy:.4f}")
        if strategy_name != 'baseline':
            print(f"  平均重复次数: {avg_repeats:.2f}")

        # 保存指标历史（便于对比曲线）
        if val_loader is not None:
            strategy.save_metrics_with_accuracy(epoch, avg_loss, avg_rmse, avg_mae, avg_accuracy,
                                               val_loss, val_rmse, val_mae, val_accuracy)
        else:
            strategy.save_metrics_with_accuracy(epoch, avg_loss, avg_rmse, avg_mae, avg_accuracy,
                                               test_loss, test_rmse, test_mae, test_accuracy)

        # 早停（使用验证集上的 Loss；若无验证集则使用测试集）
        if args.early_stopping:
            current_val = val_loss if val_loader is not None else test_loss
            if current_val + 1e-8 < best_val_metric:
                best_val_metric = current_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"早停触发（patience={args.patience}）。提前结束 {strategy_name} 的训练。")
                    break
    
    total_time = time.time() - start_time
    print(f"{strategy_name} 策略训练完成! 总时间: {total_time:.2f}秒")
    
    # 保存指标历史
    save_path = os.path.join(args.save_dir, f'{strategy_name}_metrics.npy')
    np.save(save_path, strategy.metrics_history)
    
    return strategy

def train(args):
    # 若指定预设，优先覆盖参数
    apply_preset(args)
    # 规范化保存目录：若为相对路径，则相对本脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(script_dir, args.save_dir)
    # 创建结果保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建唯一的TensorBoard日志目录（保留历史，不再清空整个目录）
    tensorboard_base_dir = os.path.join(args.save_dir, 'tensorboard')
    os.makedirs(tensorboard_base_dir, exist_ok=True)
    tensorboard_dir = os.path.join(tensorboard_base_dir, f"{args.dataset}_{int(time.time())}")
    os.makedirs(tensorboard_dir, exist_ok=True)

    # 加载数据
    data_loader = HousingDataLoader(dataset_name=args.dataset, batch_size=args.batch_size, val_ratio=args.val_ratio)
    
    # 获取输入维度
    input_dim = next(iter(data_loader.get_train_loader()))[0].shape[1]
    
    # 训练所有指定的策略：为每种策略创建单独的 TensorBoard 子目录
    strategies = {}
    for strategy_name in args.strategies:
        run_dir = os.path.join(tensorboard_dir, strategy_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
        strategies[strategy_name] = train_single_strategy(strategy_name, args, data_loader, input_dim, writer)
        writer.close()

if __name__ == '__main__':
    args = parse_args()
    train(args)