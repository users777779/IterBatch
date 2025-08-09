"""统一训练脚本

功能：
- 解析 CLI 参数，构建数据、模型与训练策略
- 训练循环：逐批训练 + 逐批 TensorBoard 写入
- 每 epoch 评价 Train/Val/Test，记录标量到 TensorBoard
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
import shutil
from model import MLP
from data_loader import HousingDataLoader
from strategy import BaselineStrategy, ABRStrategy, LearnableSchedulingStrategy, SlidingWindowStrategy
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='回归任务统一实验')
    parser.add_argument('--dataset', type=str, default='california', choices=['california', 'boston'],
                        help='数据集选择 (california/boston)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'],
                        help='模型选择 (当前仅支持mlp)')
    parser.add_argument('--strategies', type=str, nargs='+', default=['baseline'], 
                        choices=['baseline', 'abr', 'learnable', 'window'],
                        help='训练策略选择 (baseline/abr/learnable/window)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--loss_threshold', type=float, default=0.3, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=5, help='ABR策略的最大重复次数')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口策略的窗口大小')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='验证集比例（0~0.5），用于早停与超参选择')
    parser.add_argument('--early_stopping', action='store_true', help='是否启用早停（基于验证集或测试集）')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值（若指标无提升的最大连续 epoch 数）')
    # Learnable 策略增强参数
    parser.add_argument('--epsilon', type=float, default=0.1, help='Learnable 策略的探索率 ε（ε-greedy）')
    parser.add_argument('--ema_gamma', type=float, default=0.8, help='Learnable 输出重复次数的 EMA 平滑系数 γ')
    parser.add_argument('--budget_ratio', type=float, default=None, help='Learnable 额外重复次数预算比例（相对 baseline，每 epoch）例如 0.5/1/2；None 表示不限制')
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
        )
    elif strategy_name == 'window':
        return SlidingWindowStrategy(model, criterion, optimizer, args.window_size, args.loss_threshold, args.max_repeats)
    else:
        raise ValueError(f"不支持的策略: {strategy_name}")

def calculate_metrics(outputs, targets):
    """计算评估指标"""
    loss = torch.mean((outputs - targets) ** 2)
    rmse = torch.sqrt(loss)
    mae = torch.mean(torch.abs(outputs - targets))
    # 计算accuracy: 预测值与真实值之差小于0.1的比例
    accuracy_threshold = 0.1
    accuracy = torch.mean((torch.abs(outputs - targets) < accuracy_threshold).float())
    return loss.item(), rmse.item(), mae.item(), accuracy.item()


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
        epoch_start_time = time.time()
        total_loss = 0
        total_rmse = 0
        total_mae = 0
        total_accuracy = 0
        total_repeats = 0
        count = 0
        
        # 训练一个epoch
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

            # 逐批次写入 TensorBoard（按 step 全局计数：epoch 内 step_idx）
            global_step = epoch * len(train_loader) + step_idx
            writer.add_scalar(f'{strategy_name}/Batch/Loss', batch_loss, global_step)
            writer.add_scalar(f'{strategy_name}/Batch/RMSE', batch_rmse, global_step)
            writer.add_scalar(f'{strategy_name}/Batch/MAE', batch_mae, global_step)
            writer.add_scalar(f'{strategy_name}/Batch/Accuracy', batch_accuracy, global_step)
            if strategy_name != 'baseline':
                writer.add_scalar(f'{strategy_name}/Batch/Repeats', repeats, global_step)
        
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
        
        # 计算测试集accuracy
        total_test_accuracy = 0
        test_count = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, _, _, test_accuracy = calculate_metrics(outputs, targets)
                total_test_accuracy += test_accuracy * inputs.size(0)
                test_count += inputs.size(0)
        test_accuracy = total_test_accuracy / test_count

        # 计算验证集 accuracy（若有）
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
        
        # 记录TensorBoard指标
        # 在第一个epoch记录超参数
        if epoch == 0:
            writer.add_text(f'{strategy_name}/Experiment Config', f"Strategy: {strategy_name}, Dataset: {args.dataset}, Batch Size: {args.batch_size}, Learning Rate: {args.lr}")
        
        # Loss metrics
        writer.add_scalar(f'{strategy_name}/Loss/Train', avg_loss, epoch)
        writer.add_scalar(f'{strategy_name}/Loss/Val', val_loss, epoch)
        writer.add_scalar(f'{strategy_name}/Loss/Test', test_loss, epoch)
        
        # RMSE metrics
        writer.add_scalar(f'{strategy_name}/RMSE/Train', avg_rmse, epoch)
        writer.add_scalar(f'{strategy_name}/RMSE/Val', val_rmse, epoch)
        writer.add_scalar(f'{strategy_name}/RMSE/Test', test_rmse, epoch)
        
        # MAE metrics
        writer.add_scalar(f'{strategy_name}/MAE/Train', avg_mae, epoch)
        writer.add_scalar(f'{strategy_name}/MAE/Val', val_mae, epoch)
        writer.add_scalar(f'{strategy_name}/MAE/Test', test_mae, epoch)
        
        # Accuracy metrics
        writer.add_scalar(f'{strategy_name}/Accuracy/Train', avg_accuracy, epoch)
        if val_accuracy is not None:
            writer.add_scalar(f'{strategy_name}/Accuracy/Val', val_accuracy, epoch)
        writer.add_scalar(f'{strategy_name}/Accuracy/Test', test_accuracy, epoch)
        
        # 训练时间
        writer.add_scalar(f'{strategy_name}/Time/Epoch', epoch_time, epoch)
        
        if strategy_name != 'baseline':
            writer.add_scalar(f'{strategy_name}/Repeats/Avg', avg_repeats, epoch)
            
        print(f"{strategy_name} - Epoch {epoch+1}/{args.epochs}, 训练时间: {epoch_time:.2f}秒")
        print(f"  训练损失: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, Accuracy: {avg_accuracy:.4f}")
        if val_loader is not None:
            print(f"  验证损失: {val_loss:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"  测试损失: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, Accuracy: {test_accuracy:.4f}")
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
    # 创建结果保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建唯一的TensorBoard日志目录
    tensorboard_base_dir = os.path.join(args.save_dir, 'tensorboard')
    tensorboard_dir = os.path.join(tensorboard_base_dir, f"{args.dataset}_{int(time.time())}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 清理之前的日志文件
    if os.path.exists(tensorboard_base_dir):
        for item in os.listdir(tensorboard_base_dir):
            item_path = os.path.join(tensorboard_base_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # 加载数据
    data_loader = HousingDataLoader(dataset_name=args.dataset, batch_size=args.batch_size, val_ratio=args.val_ratio)
    
    # 获取输入维度
    input_dim = next(iter(data_loader.get_train_loader()))[0].shape[1]
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # 训练所有指定的策略
    strategies = {}
    for strategy_name in args.strategies:
        strategies[strategy_name] = train_single_strategy(strategy_name, args, data_loader, input_dim, writer)
    
    # 关闭TensorBoard writer
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    train(args)