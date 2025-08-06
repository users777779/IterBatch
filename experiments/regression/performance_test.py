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

# Removed Chinese font settings

def parse_args():    
    parser = argparse.ArgumentParser(description='Regression Task Performance Test')
    parser.add_argument('--dataset', type=str, default='california', choices=['california', 'boston'],
                        help='Dataset selection (california/boston)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--loss_threshold', type=float, default=0.5, help='Loss threshold for ABR strategy')
    parser.add_argument('--max_repeats', type=int, default=5, help='Maximum repeats for ABR strategy')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for sliding window strategy')
    parser.add_argument('--save_dir', type=str, default='performance_results', help='Directory to save results')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per strategy (average will be taken)')
    return parser.parse_args()


def test_strategy(strategy_name, model, criterion, optimizer, train_loader, test_loader, args, scheduler_model=None, scheduler_optimizer=None):
    """
    Test the performance of a single strategy
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
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    # 训练循环
    start_time = time.time()
    total_repeats = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_repeats = 0

        for inputs, targets in tqdm(train_loader, desc=f"{strategy_name} Strategy Epoch {epoch+1}/{args.epochs}"):
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
    print('Loading dataset...')
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
        print(f"\nTesting {strategy_name} strategy...")
        strategy_results = []

        for run in range(1, args.num_runs + 1):
            print(f"  Run {run}/{args.num_runs}")

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
        print(f"  Average training time: {avg_result['avg_total_time']:.2f} ± {avg_result['std_total_time']:.2f}s")
        print(f"  Average repeats per batch: {avg_result['avg_repeats_per_batch']:.2f}")
        print(f"  Average test RMSE: {avg_result['avg_final_test_rmse']:.4f} ± {avg_result['std_final_test_rmse']:.4f}")
        print(f"  Average test MAE: {avg_result['avg_final_test_mae']:.4f} ± {avg_result['std_final_test_mae']:.4f}")

        # 保存指标历史
        for i, result in enumerate(strategy_results):
            np.save(os.path.join(args.save_dir, f'{strategy_name}_metrics_run_{i+1}.npy'), result['metrics_history'])

    # 保存所有结果到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.save_dir, 'performance_comparison.csv'), index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {args.save_dir}/performance_comparison.csv")

    # 可视化比较结果
    visualize_results(all_results, args.save_dir)


def visualize_results(results, save_dir):
    """
    Visualize performance comparison of different strategies
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
    plt.xlabel('Training Strategy')
    plt.ylabel('Average Training Time (s)')
    plt.title('Training Time Comparison of Different Strategies')
    plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'))

    # 绘制RMSE比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_final_test_rmse, yerr=[r['std_final_test_rmse'] for r in results], capsize=5)
    plt.xlabel('Training Strategy')
    plt.ylabel('Average Test RMSE')
    plt.title('Test RMSE Comparison of Different Strategies')
    plt.savefig(os.path.join(save_dir, 'rmse_comparison.png'))

    # 绘制MAE比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_final_test_mae, yerr=[r['std_final_test_mae'] for r in results], capsize=5)
    plt.xlabel('Training Strategy')
    plt.ylabel('Average Test MAE')
    plt.title('Test MAE Comparison of Different Strategies')
    plt.savefig(os.path.join(save_dir, 'mae_comparison.png'))

    # 绘制重复次数比较
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, avg_repeats)
    plt.xlabel('Training Strategy')
    plt.ylabel('Average Repeats per Batch')
    plt.title('Repetition Comparison of Different Strategies')
    plt.savefig(os.path.join(save_dir, 'repeats_comparison.png'))

    print(f"Visualization charts saved to {save_dir} directory")


if __name__ == '__main__':
    args = parse_args()
    run_tests(args)