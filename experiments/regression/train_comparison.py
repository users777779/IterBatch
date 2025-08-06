import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from model import MLP
from data_loader import HousingDataLoader
from strategy import BaselineStrategy, ABRStrategy, LearnableSchedulingStrategy, SlidingWindowStrategy

# 使用默认字体以避免中文显示问题

def parse_args():
    parser = argparse.ArgumentParser(description='回归任务对比实验')
    parser.add_argument('--dataset', type=str, default='california', choices=['california', 'boston'],
                        help='数据集选择 (california/boston)')
    parser.add_argument('--batch_size', type=int, default=5, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--loss_threshold', type=float, default=0.5, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=1, help='ABR策略的最大重复次数')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口策略的窗口大小')
    parser.add_argument('--save_dir', type=str, default='/root/IterBatch/experiments/regression/results_comparison', help='结果保存目录')
    return parser.parse_args()


def train(args):
    # 创建结果保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(args.save_dir, 'experiment_log.txt')
    with open(log_file, 'w') as f:
        f.write(f'Experiment started at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Learning rate: {args.lr}\n')
        f.write(f'Hidden dimension: {args.hidden_dim}\n')
        f.write('\n')

    # 加载数据
    print('Loading dataset...')
    data_loader = HousingDataLoader(dataset_name=args.dataset, batch_size=args.batch_size)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 记录数据信息
    with open(log_file, 'a') as f:
        f.write(f'Dataset loaded at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Training samples: {len(train_loader.dataset)}\n')
        f.write(f'Test samples: {len(test_loader.dataset)}\n')
        f.write(f'Number of training batches: {len(train_loader)}\n')
        f.write(f'Number of test batches: {len(test_loader)}\n')
        f.write('\n')

    # 获取输入维度
    input_dim = next(iter(train_loader))[0].shape[1]

    # 创建主模型
    main_model = MLP(input_dim, args.hidden_dim)

    # 创建四个克隆体模型
    model_base = main_model.clone()
    model_abr = main_model.clone()
    model_learnable = main_model.clone()
    model_window = main_model.clone()

    # 创建优化器和损失函数
    criterion = nn.MSELoss()

    optimizer_base = optim.Adam(model_base.parameters(), lr=args.lr)
    optimizer_abr = optim.Adam(model_abr.parameters(), lr=args.lr)
    optimizer_learnable = optim.Adam(model_learnable.parameters(), lr=args.lr)
    optimizer_window = optim.Adam(model_window.parameters(), lr=args.lr)

    # 创建调度模型（用于可学习调度策略）
    scheduler_model = MLP(1, 32, 1)
    scheduler_optimizer = optim.Adam(scheduler_model.parameters(), lr=args.lr)

    # 创建训练策略
    strategy_base = BaselineStrategy(model_base, criterion, optimizer_base)
    strategy_abr = ABRStrategy(model_abr, criterion, optimizer_abr, args.loss_threshold, args.max_repeats)
    strategy_learnable = LearnableSchedulingStrategy(model_learnable, criterion, optimizer_learnable, scheduler_model, scheduler_optimizer)
    strategy_window = SlidingWindowStrategy(model_window, criterion, optimizer_window, args.window_size, args.loss_threshold, args.max_repeats)

    # 训练循环
    print("开始对比实验训练...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # 每轮epoch开始时，重置所有模型的状态
        model_base.train()
        model_abr.train()
        model_learnable.train()

        total_loss_base = 0
        total_loss_abr = 0
        total_loss_learnable = 0
        total_loss_window = 0
        total_rmse_base = 0
        total_rmse_abr = 0
        total_rmse_learnable = 0
        total_rmse_window = 0
        total_mae_base = 0
        total_mae_abr = 0
        total_mae_learnable = 0
        total_mae_window = 0
        total_repeats_abr = 0
        total_repeats_learnable = 0
        total_repeats_window = 0
        count = 0

        # 训练一个epoch
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 记录批次信息
            if batch_idx % 10 == 0:  # 每10个批次记录一次
                with open(log_file, 'a') as f:
                    f.write(f'Epoch {epoch+1}, Batch {batch_idx}\n')
                    f.write(f'Input shape: {inputs.shape}\n')
                    f.write(f'Target shape: {targets.shape}\n')
                    f.write(f'Target min: {targets.min().item()}, max: {targets.max().item()}, mean: {targets.mean().item()}\n')
                    f.write('\n')
            # 基线策略训练
            loss_base, repeats_base = strategy_base.train_batch(inputs, targets)
            outputs_base = model_base(inputs)
            rmse_base = torch.sqrt(torch.mean((outputs_base - targets) ** 2))
            mae_base = torch.mean(torch.abs(outputs_base - targets))

            # ABR策略训练
            loss_abr, repeats_abr = strategy_abr.train_batch(inputs, targets)
            outputs_abr = model_abr(inputs)
            rmse_abr = torch.sqrt(torch.mean((outputs_abr - targets) ** 2))
            mae_abr = torch.mean(torch.abs(outputs_abr - targets))

            # 可学习调度策略训练
            loss_learnable, repeats_learnable = strategy_learnable.train_batch(inputs, targets)
            outputs_learnable = model_learnable(inputs)
            rmse_learnable = torch.sqrt(torch.mean((outputs_learnable - targets) ** 2))
            mae_learnable = torch.mean(torch.abs(outputs_learnable - targets))

            # 滑动窗口策略训练
            loss_window, repeats_window = strategy_window.train_batch(inputs, targets)
            outputs_window = model_window(inputs)
            rmse_window = torch.sqrt(torch.mean((outputs_window - targets) ** 2))
            mae_window = torch.mean(torch.abs(outputs_window - targets))

            # 累加指标
            batch_size = inputs.size(0)
            total_loss_base += loss_base * batch_size
            total_loss_abr += loss_abr * batch_size
            total_loss_learnable += loss_learnable * batch_size
            total_loss_window += loss_window * batch_size
            total_rmse_base += rmse_base.item() * batch_size
            total_rmse_abr += rmse_abr.item() * batch_size
            total_rmse_learnable += rmse_learnable.item() * batch_size
            total_rmse_window += rmse_window.item() * batch_size
            total_mae_base += mae_base.item() * batch_size
            total_mae_abr += mae_abr.item() * batch_size
            total_mae_learnable += mae_learnable.item() * batch_size
            total_mae_window += mae_window.item() * batch_size
            total_repeats_abr += repeats_abr
            total_repeats_learnable += repeats_learnable
            total_repeats_window += repeats_window
            count += batch_size

        # 计算平均指标
        avg_loss_base = total_loss_base / count
        avg_loss_abr = total_loss_abr / count
        avg_loss_learnable = total_loss_learnable / count
        avg_loss_window = total_loss_window / count
        avg_rmse_base = total_rmse_base / count
        avg_rmse_abr = total_rmse_abr / count
        avg_rmse_learnable = total_rmse_learnable / count
        avg_rmse_window = total_rmse_window / count
        avg_mae_base = total_mae_base / count
        avg_mae_abr = total_mae_abr / count
        avg_mae_learnable = total_mae_learnable / count
        avg_mae_window = total_mae_window / count
        avg_repeats_abr = total_repeats_abr / len(train_loader)
        avg_repeats_learnable = total_repeats_learnable / len(train_loader)
        avg_repeats_window = total_repeats_window / len(train_loader)

        # 在测试集上评估
        test_loss_base, test_rmse_base, test_mae_base = strategy_base.evaluate(test_loader)
        test_loss_abr, test_rmse_abr, test_mae_abr = strategy_abr.evaluate(test_loader)
        test_loss_learnable, test_rmse_learnable, test_mae_learnable = strategy_learnable.evaluate(test_loader)
        test_loss_window, test_rmse_window, test_mae_window = strategy_window.evaluate(test_loader)

        # 保存指标
        strategy_base.save_metrics(epoch+1, avg_loss_base, avg_rmse_base, avg_mae_base, test_loss_base, test_rmse_base, test_mae_base)
        strategy_abr.save_metrics(epoch+1, avg_loss_abr, avg_rmse_abr, avg_mae_abr, test_loss_abr, test_rmse_abr, test_mae_abr)
        strategy_learnable.save_metrics(epoch+1, avg_loss_learnable, avg_rmse_learnable, avg_mae_learnable, test_loss_learnable, test_rmse_learnable, test_mae_learnable)
        strategy_window.save_metrics(epoch+1, avg_loss_window, avg_rmse_window, avg_mae_window, test_loss_window, test_rmse_window, test_mae_window)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs}, 训练时间: {epoch_time:.2f}秒")
        print(f"  基线策略 - 训练损失: {avg_loss_base:.4f}, RMSE: {avg_rmse_base:.4f}, MAE: {avg_mae_base:.4f}")
        print(f"  ABR策略   - 训练损失: {avg_loss_abr:.4f}, RMSE: {avg_rmse_abr:.4f}, MAE: {avg_mae_abr:.4f}, 平均重复次数: {avg_repeats_abr:.2f}")
        print(f"  可学习策略 - 训练损失: {avg_loss_learnable:.4f}, RMSE: {avg_rmse_learnable:.4f}, MAE: {avg_mae_learnable:.4f}, 平均重复次数: {avg_repeats_learnable:.2f}")
        print(f"  滑动窗口策略 - 训练损失: {avg_loss_window:.4f}, RMSE: {avg_rmse_window:.4f}, MAE: {avg_mae_window:.4f}, 平均重复次数: {avg_repeats_window:.2f}")
        print(f"  基线策略 - 测试损失: {test_loss_base:.4f}, RMSE: {test_rmse_base:.4f}, MAE: {test_mae_base:.4f}")
        print(f"  ABR策略   - 测试损失: {test_loss_abr:.4f}, RMSE: {test_rmse_abr:.4f}, MAE: {test_mae_abr:.4f}")
        print(f"  可学习策略 - 测试损失: {test_loss_learnable:.4f}, RMSE: {test_rmse_learnable:.4f}, MAE: {test_mae_learnable:.4f}")
        print(f"  滑动窗口策略 - 测试损失: {test_loss_window:.4f}, RMSE: {test_rmse_window:.4f}, MAE: {test_mae_window:.4f}")

    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time:.2f}秒")

    # 保存指标历史
    np.save(os.path.join(args.save_dir, 'baseline_metrics.npy'), strategy_base.metrics_history)
    np.save(os.path.join(args.save_dir, 'abr_metrics.npy'), strategy_abr.metrics_history)
    np.save(os.path.join(args.save_dir, 'learnable_metrics.npy'), strategy_learnable.metrics_history)
    np.save(os.path.join(args.save_dir, 'window_metrics.npy'), strategy_window.metrics_history)

    # 绘制损失曲线对比
    plt.figure(figsize=(12, 7))
    plt.plot(strategy_base.metrics_history['epoch'], strategy_base.metrics_history['train_loss'], label='Baseline-Train')
    plt.plot(strategy_abr.metrics_history['epoch'], strategy_abr.metrics_history['train_loss'], label='ABR-Train')
    plt.plot(strategy_learnable.metrics_history['epoch'], strategy_learnable.metrics_history['train_loss'], label='Learnable-Train')
    plt.plot(strategy_window.metrics_history['epoch'], strategy_window.metrics_history['train_loss'], label='SlidingWindow-Train')
    plt.plot(strategy_base.metrics_history['epoch'], strategy_base.metrics_history['test_loss'], '--', label='Baseline-Test')
    plt.plot(strategy_abr.metrics_history['epoch'], strategy_abr.metrics_history['test_loss'], '--', label='ABR-Test')
    plt.plot(strategy_learnable.metrics_history['epoch'], strategy_learnable.metrics_history['test_loss'], '--', label='Learnable-Test')
    plt.plot(strategy_window.metrics_history['epoch'], strategy_window.metrics_history['test_loss'], '--', label='SlidingWindow-Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve Comparison of Different Strategies')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'loss_comparison.png'))

    # 绘制RMSE曲线对比
    plt.figure(figsize=(12, 7))
    plt.plot(strategy_base.metrics_history['epoch'], strategy_base.metrics_history['train_rmse'], label='Baseline-Train')
    plt.plot(strategy_abr.metrics_history['epoch'], strategy_abr.metrics_history['train_rmse'], label='ABR-Train')
    plt.plot(strategy_learnable.metrics_history['epoch'], strategy_learnable.metrics_history['train_rmse'], label='Learnable-Train')
    plt.plot(strategy_window.metrics_history['epoch'], strategy_window.metrics_history['train_rmse'], label='SlidingWindow-Train')
    plt.plot(strategy_base.metrics_history['epoch'], strategy_base.metrics_history['test_rmse'], '--', label='Baseline-Test')
    plt.plot(strategy_abr.metrics_history['epoch'], strategy_abr.metrics_history['test_rmse'], '--', label='ABR-Test')
    plt.plot(strategy_learnable.metrics_history['epoch'], strategy_learnable.metrics_history['test_rmse'], '--', label='Learnable-Test')
    plt.plot(strategy_window.metrics_history['epoch'], strategy_window.metrics_history['test_rmse'], '--', label='SlidingWindow-Test')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Curve Comparison of Different Strategies')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'rmse_comparison.png'))

    # 绘制MAE曲线对比
    plt.figure(figsize=(12, 7))
    plt.plot(strategy_base.metrics_history['epoch'], strategy_base.metrics_history['train_mae'], label='Baseline-Train')
    plt.plot(strategy_abr.metrics_history['epoch'], strategy_abr.metrics_history['train_mae'], label='ABR-Train')
    plt.plot(strategy_learnable.metrics_history['epoch'], strategy_learnable.metrics_history['train_mae'], label='Learnable-Train')
    plt.plot(strategy_window.metrics_history['epoch'], strategy_window.metrics_history['train_mae'], label='SlidingWindow-Train')
    plt.plot(strategy_base.metrics_history['epoch'], strategy_base.metrics_history['test_mae'], '--', label='Baseline-Test')
    plt.plot(strategy_abr.metrics_history['epoch'], strategy_abr.metrics_history['test_mae'], '--', label='ABR-Test')
    plt.plot(strategy_learnable.metrics_history['epoch'], strategy_learnable.metrics_history['test_mae'], '--', label='Learnable-Test')
    plt.plot(strategy_window.metrics_history['epoch'], strategy_window.metrics_history['test_mae'], '--', label='SlidingWindow-Test')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE Curve Comparison of Different Strategies')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'mae_comparison.png'))

    # 绘制滑动窗口策略的窗口趋势图
    if len(strategy_window.batch_history) > 0:
        plt.figure(figsize=(10, 6))
        trends = [item['window_trend'] for item in strategy_window.batch_history]
        plt.plot(range(len(trends)), trends)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Batch Index')
        plt.ylabel('Window Loss Trend')
        plt.title('Sliding Window Strategy: Loss Trend Change')
        plt.savefig(os.path.join(args.save_dir, 'window_trend.png'))

if __name__ == '__main__':
    args = parse_args()
    train(args)