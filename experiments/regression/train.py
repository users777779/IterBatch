import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from model import MLP
from data_loader import HousingDataLoader
from strategy import BaselineStrategy, ABRStrategy, LearnableSchedulingStrategy

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_args():
    parser = argparse.ArgumentParser(description='回归任务实验')
    parser.add_argument('--dataset', type=str, default='california', choices=['california', 'boston'],
                        help='数据集选择 (california/boston)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'],
                        help='模型选择 (当前仅支持mlp)')
    parser.add_argument('--strategy', type=str, default='abr', choices=['baseline', 'abr', 'learnable'],
                        help='训练策略选择 (baseline/abr/learnable)')
    parser.add_argument('--batch_size', type=int, default=5, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--loss_threshold', type=float, default=0.5, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=5, help='ABR策略的最大重复次数')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    return parser.parse_args()


def train(args):
    # 创建结果保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    data_loader = HousingDataLoader(dataset_name=args.dataset, batch_size=args.batch_size)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 获取输入维度
    input_dim = next(iter(train_loader))[0].shape[1]

    # 创建模型
    if args.model == 'mlp':
        model = MLP(input_dim, args.hidden_dim)
    else:
        raise ValueError(f"不支持的模型: {args.model}")

    # 创建优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建训练策略
    if args.strategy == 'baseline':
        strategy = BaselineStrategy(model, criterion, optimizer)
    elif args.strategy == 'abr':
        strategy = ABRStrategy(model, criterion, optimizer, args.loss_threshold, args.max_repeats)
    elif args.strategy == 'learnable':
        # 创建调度模型（简单的MLP）
        scheduler_model = MLP(1, 32, 1)
        scheduler_optimizer = optim.Adam(scheduler_model.parameters(), lr=args.lr)
        strategy = LearnableSchedulingStrategy(model, criterion, optimizer, scheduler_model, scheduler_optimizer)
    else:
        raise ValueError(f"不支持的策略: {args.strategy}")

    # 训练循环
    print(f"开始训练 {args.strategy} 策略...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0
        total_rmse = 0
        total_mae = 0
        total_repeats = 0
        count = 0

        # 训练一个epoch
        for inputs, targets in train_loader:
            loss, repeats = strategy.train_batch(inputs, targets)
            outputs = model(inputs)
            rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))
            mae = torch.mean(torch.abs(outputs - targets))

            total_loss += loss * inputs.size(0)
            total_rmse += rmse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_repeats += repeats
            count += inputs.size(0)

        avg_loss = total_loss / count
        avg_rmse = total_rmse / count
        avg_mae = total_mae / count
        avg_repeats = total_repeats / len(train_loader)

        # 在测试集上评估
        test_loss, test_rmse, test_mae = strategy.evaluate(test_loader)

        # 保存指标
        strategy.save_metrics(epoch+1, avg_loss, avg_rmse, avg_mae, test_loss, test_rmse, test_mae)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs}, 训练时间: {epoch_time:.2f}秒")
        print(f"  训练损失: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}")
        print(f"  测试损失: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        if args.strategy != 'baseline':
            print(f"  平均重复次数: {avg_repeats:.2f}")

    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time:.2f}秒")

    # 保存指标历史
    np.save(os.path.join(args.save_dir, f'{args.strategy}_metrics.npy'), strategy.metrics_history)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(strategy.metrics_history['epoch'], strategy.metrics_history['train_loss'], label='训练损失')
    plt.plot(strategy.metrics_history['epoch'], strategy.metrics_history['test_loss'], label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title(f'{args.strategy} 策略损失曲线')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.strategy}_loss_curve.png'))

    # 绘制RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(strategy.metrics_history['epoch'], strategy.metrics_history['train_rmse'], label='训练RMSE')
    plt.plot(strategy.metrics_history['epoch'], strategy.metrics_history['test_rmse'], label='测试RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'{args.strategy} 策略RMSE曲线')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.strategy}_rmse_curve.png'))

    # 如果是ABR策略，绘制批次历史分析
    if args.strategy == 'abr':
        initial_losses = [item['initial_loss'] for item in strategy.batch_history]
        repeat_counts = [item['repeat_count'] for item in strategy.batch_history]

        plt.figure(figsize=(10, 6))
        plt.scatter(initial_losses, repeat_counts)
        plt.xlabel('初始损失')
        plt.ylabel('重复次数')
        plt.title('ABR策略：初始损失与重复次数关系')
        plt.savefig(os.path.join(args.save_dir, 'abr_repeat_analysis.png'))

if __name__ == '__main__':
    import time
    args = parse_args()
    train(args)