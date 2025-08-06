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
import pandas as pd
from tqdm import tqdm

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_args():    
    parser = argparse.ArgumentParser(description='回归任务性能测试')
    parser.add_argument('--dataset', type=str, default='california', choices=['california', 'boston'],
                        help='数据集选择 (california/boston)')
    parser.add_argument('--batch_size', type=int, default=5, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--loss_threshold', type=float, default=0.5, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=5, help='ABR策略的最大重复次数')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口策略的窗口大小')
    parser.add_argument('--save_dir', type=str, default='performance_results', help='结果保存目录')
    parser.add_argument('--num_runs', type=int, default=3, help='每种策略运行的次数，取平均值')
    return parser.parse_args()


def test_strategy(strategy_name, model, criterion, optimizer, train_loader, test_loader, args, scheduler_model=None, scheduler_optimizer=None):
    """
    测试单一策略的性能
    """
    # 创建训练策略
    if strategy_name == 'baseline':
        strategy = BaselineStrategy(model, criterion, optimizer)
    elif strategy_name == 'abr':
        strategy = ABRStrategy(model, criterion, optimizer, args.loss_threshold, args.max_repeats)
    elif strategy_name == 'learnable':
        strategy = LearnableSchedulingStrategy(model, criterion, optimizer, scheduler_model, scheduler_optimizer)
    elif strategy_name == 'window':
        strategy = SlidingWindowStrategy(model, criterion, optimizer, args.window_size, args.loss_threshold, args.max_repeats)
    else:
        raise ValueError(f"不支持的策略: {strategy_name}")

    # 训练循环
    start_time = time.time()
    total_repeats = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_repeats = 0

        for inputs, targets in tqdm(train_loader, desc=f"{strategy_name} 策略 Epoch {epoch+1}/{args.epochs}"):
            loss, repeats = strategy.train_batch(inputs, targets)
            epoch_repeats += repeats

        total_repeats += epoch_repeats / len(train_loader)

        # 在测试集上评估
        test_loss, test_rmse, test_mae = strategy.evaluate(test_loader)
        strategy.save_metrics(epoch+1, 0, 0, 0, test_loss, test_rmse, test_mae)  # 简化，只记录测试指标

    total_time = time.time() - start_time
    avg_repeats = total_repeats / args.epochs

    # 获取最终测试指标
    final_test_rmse = strategy.metrics_history['test_rmse'][-1]
    final_test_mae = strategy.metrics_history['test_mae'][-1]

    return {
        'strategy': strategy_name,
        'total_time': total_time,
        'avg_repeats_per_batch': avg_repeats,
        'final_test_rmse': final_test_rmse,
        'final_test_mae': final_test_mae,
        'metrics_history': strategy.metrics_history
    }


def run_tests(args):
    # 创建结果保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    print('加载数据集...')
    data_loader = HousingDataLoader(dataset_name=args.dataset, batch_size=args.batch_size)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 获取输入维度
    input_dim = next(iter(train_loader))[0].shape[1]

    # 定义结果列表
    all_results = []

    # 测试所有策略
    strategies = ['baseline', 'abr', 'learnable', 'window']

    for strategy_name in strategies:
        print(f"\n测试 {strategy_name} 策略...")
        strategy_results = []

        for run in range(1, args.num_runs + 1):
            print(f"  运行 {run}/{args.num_runs}")

            # 创建新模型
            model = MLP(input_dim, args.hidden_dim)

            # 创建优化器和损失函数
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # 特殊处理可学习调度策略
            scheduler_model = None
            scheduler_optimizer = None
            if strategy_name == 'learnable':
                scheduler_model = MLP(1, 32, 1)
                scheduler_optimizer = optim.Adam(scheduler_model.parameters(), lr=args.lr)

            # 测试策略
            result = test_strategy(
                strategy_name, model, criterion, optimizer, train_loader, test_loader, args,
                scheduler_model, scheduler_optimizer
            )
            strategy_results.append(result)

        # 计算平均结果
        avg_result = {
            'strategy': strategy_name,
            'avg_total_time': np.mean([r['total_time'] for r in strategy_results]),
            'avg_repeats_per_batch': np.mean([r['avg_repeats_per_batch'] for r in strategy_results]),
            'avg_final_test_rmse': np.mean([r['final_test_rmse'] for r in strategy_results]),
            'avg_final_test_mae': np.mean([r['final_test_mae'] for r in strategy_results]),
            'std_total_time': np.std([r['total_time'] for r in strategy_results]),
            'std_final_test_rmse': np.std([r['final_test_rmse'] for r in strategy_results]),
            'std_final_test_mae': np.std([r['final_test_mae'] for r in strategy_results])
        }

        all_results.append(avg_result)
        print(f"  平均训练时间: {avg_result['avg_total_time']:.2f} ± {avg_result['std_total_time']:.2f}秒")
        print(f"  平均重复次数/批次: {avg_result['avg_repeats_per_batch']:.2f}")
        print(f"  平均测试RMSE: {avg_result['avg_final_test_rmse']:.4f} ± {avg_result['std_final_test_rmse']:.4f}")
        print(f"  平均测试MAE: {avg_result['avg_final_test_mae']:.4f} ± {avg_result['std_final_test_mae']:.4f}")

        # 保存指标历史
        for i, result in enumerate(strategy_results):
            np.save(os.path.join(args.save_dir, f'{strategy_name}_metrics_run_{i+1}.npy'), result['metrics_history'])

    # 保存所有结果到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.save_dir, 'performance_comparison.csv'), index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 {args.save_dir}/performance_comparison.csv")

    # 可视化比较结果
    visualize_results(all_results, args.save_dir)


def visualize_results(results, save_dir):
    """
    可视化不同策略的性能比较
    """
    # 提取数据
    strategies = [r['strategy'] for r in results]
    avg_total_time = [r['avg_total_time'] for r in results]
    avg_final_test_rmse = [r['avg_final_test_rmse'] for r in results]
    avg_final_test_mae = [r['avg_final_test_mae'] for r in results]
    avg_repeats = [r['avg_repeats_per_batch'] for r in results]

    # 绘制训练时间比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_total_time, yerr=[r['std_total_time'] for r in results], capsize=5)
    plt.xlabel('训练策略')
    plt.ylabel('平均训练时间 (秒)')
    plt.title('不同策略的训练时间比较')
    plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'))

    # 绘制RMSE比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_final_test_rmse, yerr=[r['std_final_test_rmse'] for r in results], capsize=5)
    plt.xlabel('训练策略')
    plt.ylabel('平均测试RMSE')
    plt.title('不同策略的测试RMSE比较')
    plt.savefig(os.path.join(save_dir, 'rmse_comparison.png'))

    # 绘制MAE比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_final_test_mae, yerr=[r['std_final_test_mae'] for r in results], capsize=5)
    plt.xlabel('训练策略')
    plt.ylabel('平均测试MAE')
    plt.title('不同策略的测试MAE比较')
    plt.savefig(os.path.join(save_dir, 'mae_comparison.png'))

    # 绘制重复次数比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_repeats)
    plt.xlabel('训练策略')
    plt.ylabel('平均重复次数/批次')
    plt.title('不同策略的重复训练次数比较')
    plt.savefig(os.path.join(save_dir, 'repeats_comparison.png'))

    print(f"可视化图表已保存到 {save_dir} 目录")


if __name__ == '__main__':
    args = parse_args()
    run_tests(args)