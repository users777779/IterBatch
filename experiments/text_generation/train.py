import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import time
import shutil
from model import LLaMAWithLoRA, SchedulerLLaMA, WindowPolicyMLP
from data_loader import TextDataLoader
from strategy import (
    BaselineStrategy,
    ABRStrategy,
    LearnableSchedulingStrategy,
    SlidingWindowStrategy,
    LearnableWindowStrategy,
)
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='文本生成任务统一实验')
    parser.add_argument('--model', type=str, default='llama', choices=['llama'],
                        help='模型选择 (当前仅支持llama)')
    parser.add_argument('--model_name', type=str, default='huggyllama/llama-7b',
                        help='预训练模型名称或路径')
    parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'dialogue'],
                        help='数据集选择 (wikitext/dialogue)')
    parser.add_argument('--strategies', type=str, nargs='+', default=['baseline'], 
                        choices=['baseline', 'abr', 'learnable', 'window', 'lwindow'],
                        help='训练策略选择 (baseline/abr/learnable/window/lwindow)')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA矩阵的秩')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA缩放因子')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA层的dropout率')
    parser.add_argument('--loss_threshold', type=float, default=1.0, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=5, help='ABR策略的最大重复次数')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口策略的窗口大小')
    parser.add_argument('--volatility_mode', type=str, default='suppress', choices=['suppress', 'encourage'], help='滑动窗口：高波动处理方式')
    parser.add_argument('--trend_threshold', type=float, default=0.01, help='滑动窗口：趋势阈值（标准化后）')
    parser.add_argument('--vol_threshold', type=float, default=0.1, help='滑动窗口：波动阈值（标准差阈值）')
    parser.add_argument('--window_min_size', type=int, default=3, help='滑动窗口：计算趋势/方差的最小窗口长度')
    parser.add_argument('--weight_trend', type=float, default=1.0, help='滑动窗口：风险评分中趋势项权重')
    parser.add_argument('--weight_zloss', type=float, default=0.5, help='滑动窗口：风险评分中 z-loss 项权重')
    parser.add_argument('--weight_vol', type=float, default=0.5, help='滑动窗口：风险评分中波动项权重')
    parser.add_argument('--adaptive_window', action='store_true', help='是否启用基于波动的自适应窗口长度选择')
    parser.add_argument('--window_small', type=int, default=None, help='自适应窗口的较小窗口长度')
    parser.add_argument('--window_large', type=int, default=None, help='自适应窗口的较大窗口长度')
    parser.add_argument('--adapt_high_action', type=str, default='expand', choices=['expand','shrink'], help='波动高时：扩大或缩小窗口')
    parser.add_argument('--adapt_low_action', type=str, default='shrink', choices=['expand','shrink'], help='波动低时：扩大或缩小窗口')
    parser.add_argument('--vol_low_threshold', type=float, default=None, help='低波动阈值（默认 0.5*vol_threshold）')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--max_length', type=int, default=512, help='序列最大长度')
    parser.add_argument('--primer_batches', type=int, default=0, help='预热训练的批次数（使用约1000条高质量样本作为“药引子”）')
    parser.add_argument('--outlier_k', type=float, default=2.5, help='异常样本阈值系数K（mean+K*std）')
    # 新增：快速试跑控制
    parser.add_argument('--max_train_batches', type=int, default=50, help='单个epoch最多训练批次数（用于快速试跑，-1表不限制）')
    parser.add_argument('--max_eval_batches', type=int, default=50, help='验证/测试最多评估批次数（用于快速试跑，-1表不限制）')
    # 新增：步级日志记录间隔（单位：步）；设置为1表示每步记录
    parser.add_argument('--log_every_n_steps', type=int, default=1, help='每多少步记录一次训练指标到TensorBoard')
    # 可学习滑窗策略参数
    parser.add_argument('--window_policy_hidden', type=int, default=32, help='学习滑窗策略的决策网络隐藏维度（仅 lwindow 生效）')
    parser.add_argument('--policy_warmup_epochs', type=int, default=0, help='学习滑窗预热轮数（启发式 additional 监督，仅 lwindow 生效）')
    parser.add_argument('--policy_supervise_weight', type=float, default=1.0, help='学习滑窗启发式监督损失权重（仅 lwindow 生效）')
    parser.add_argument('--policy_main_weight', type=float, default=1.0, help='学习滑窗主反馈损失权重（仅 lwindow 生效）')
    parser.add_argument('--policy_epsilon', type=float, default=0.0, help='学习滑窗 ε-greedy 探索率（仅 lwindow 生效）')
    return parser.parse_args()


def get_model(model_name, model_type='llama', lora_rank=8, lora_alpha=32, lora_dropout=0.1, device=None):
    """创建模型实例"""
    if model_type == 'llama':
        model = LLaMAWithLoRA(model_name, lora_rank, lora_alpha, lora_dropout)
        # 如果是多设备映射(hf_device_map存在)或量化offload，不做整体 .to()
        if device is not None and not hasattr(model.backbone, 'hf_device_map'):
            model.to(device)
        return model
    else:
        raise ValueError(f"不支持的模型: {model_type}")


def get_strategy(model, criterion, optimizer, strategy_name, args):
    """根据策略名称创建对应的策略实例"""
    if strategy_name == 'baseline':
        return BaselineStrategy(model, criterion, optimizer)
    elif strategy_name == 'abr':
        return ABRStrategy(model, criterion, optimizer, args.loss_threshold, args.max_repeats)
    elif strategy_name == 'learnable':
        # 创建调度模型（接收loss和perplexity作为输入）
        scheduler_model = SchedulerLLaMA(32, 1)
        # 与主模型保持同设备
        scheduler_model.to(next(model.parameters()).device)
        scheduler_optimizer = optim.Adam(scheduler_model.parameters(), lr=args.lr)
        return LearnableSchedulingStrategy(model, criterion, optimizer, scheduler_model, scheduler_optimizer)
    elif strategy_name == 'window':
        return SlidingWindowStrategy(
            model,
            criterion,
            optimizer,
            window_size=args.window_size,
            loss_threshold=args.loss_threshold,
            max_repeats=args.max_repeats,
            trend_threshold=args.trend_threshold,
            vol_threshold=args.vol_threshold,
            window_min_size=args.window_min_size,
            volatility_mode=args.volatility_mode,
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
        # 引入可学习滑窗（使用独立窗口策略网络）
        policy_model = WindowPolicyMLP(hidden_dim=args.window_policy_hidden)
        policy_model.to(next(model.parameters()).device)
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


def train_single_strategy(strategy_name, args, data_loader, writer):
    """训练单个策略"""
    print(f"开始训练 {strategy_name} 策略...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = get_model(args.model_name, args.model, args.lora_rank, args.lora_alpha, args.lora_dropout, device=device)
    uses_device_map = hasattr(model.backbone, 'hf_device_map')
    
    # 创建优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建训练策略
    strategy = get_strategy(model, criterion, optimizer, strategy_name, args)
    # 配置 outlier 统计强度
    if hasattr(strategy, 'outlier_k'):
        strategy.outlier_k = float(args.outlier_k)
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()

    # 本地评估（可限制批次数）
    @torch.no_grad()
    def local_eval(model, loader, device, max_batches: int):
        model.eval()
        total_loss, count = 0.0, 0
        seen = 0
        for batch in loader:
            if uses_device_map:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            else:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
            loss_val = float(outputs.loss.item())
            total_loss += loss_val * batch['input_ids'].size(0)
            count += batch['input_ids'].size(0)
            seen += 1
            if max_batches is not None and max_batches > 0 and seen >= max_batches:
                break
        avg_loss = total_loss / max(1, count)
        ppl = float(np.exp(avg_loss))
        return avg_loss, ppl

    # 预热阶段：使用“药引子”批次进行稳定化与统计基线（不触发outlier跳过逻辑）
    if args.primer_batches > 0:
        print(f"进行预热阶段：{args.primer_batches} 个批次")
        model.train()
        seen = 0
        for batch in train_loader:
            if uses_device_map:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            else:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
            loss_val = float(outputs.loss.item())
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            if hasattr(strategy, 'loss_history'):
                strategy.loss_history.append(loss_val)
            seen += 1
            if seen >= args.primer_batches:
                break
    
    # 训练循环
    start_time = time.time()
    
    global_step = 0
    for epoch in range(args.epochs):
        # 将 epoch 信息通知策略
        if hasattr(strategy, 'set_epoch'):
            strategy.set_epoch(epoch)
        epoch_start_time = time.time()
        total_loss = 0
        total_perplexity = 0
        total_repeats = 0
        count = 0
        
        # 训练一个epoch（限制最多批次数）
        batch_seen = 0
        for step_idx, batch in enumerate(train_loader):
            # 将 step 告知策略（用于记录 outlier 的 epoch/step）
            if hasattr(strategy, 'set_step'):
                strategy.set_step(step_idx)
            loss, repeats = strategy.train_batch(batch)
            perplexity = np.exp(loss)
            
            # 步级日志
            if args.log_every_n_steps > 0 and (global_step % args.log_every_n_steps == 0):
                writer.add_scalar(f'{strategy_name}/Step/Loss', float(loss), global_step)
                writer.add_scalar(f'{strategy_name}/Step/Perplexity', float(perplexity), global_step)
                if strategy_name != 'baseline':
                    writer.add_scalar(f'{strategy_name}/Step/Repeats', float(repeats), global_step)
                # 可选记录学习率
                writer.add_scalar(f'{strategy_name}/Step/LR', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
            
            if repeats > 0:
                total_loss += loss * batch['input_ids'].size(0)
                total_perplexity += perplexity * batch['input_ids'].size(0)
                total_repeats += repeats
                count += batch['input_ids'].size(0)
                batch_seen += 1
            if args.max_train_batches is not None and args.max_train_batches > 0 and batch_seen >= args.max_train_batches:
                break
        
        avg_loss = total_loss / max(1, count)
        avg_perplexity = total_perplexity / max(1, count)
        avg_repeats = total_repeats / max(1, batch_seen)
        
        # 在验证集上评估（限制批次数）
        val_loss, val_perplexity = local_eval(model, val_loader, device, None if args.max_eval_batches == -1 else args.max_eval_batches)
        
        # 在测试集上评估（限制批次数）
        test_loss, test_perplexity = local_eval(model, test_loader, device, None if args.max_eval_batches == -1 else args.max_eval_batches)
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 记录TensorBoard指标（epoch级）
        if epoch == 0:
            writer.add_text(f'{strategy_name}/Experiment Config', f"Strategy: {strategy_name}, Dataset: {args.dataset}, Batch Size: {args.batch_size}, Learning Rate: {args.lr}")
        writer.add_scalar(f'{strategy_name}/Loss/Train', avg_loss, epoch)
        writer.add_scalar(f'{strategy_name}/Loss/Val', val_loss, epoch)
        writer.add_scalar(f'{strategy_name}/Loss/Test', test_loss, epoch)
        writer.add_scalar(f'{strategy_name}/Perplexity/Train', avg_perplexity, epoch)
        writer.add_scalar(f'{strategy_name}/Perplexity/Val', val_perplexity, epoch)
        writer.add_scalar(f'{strategy_name}/Perplexity/Test', test_perplexity, epoch)
        writer.add_scalar(f'{strategy_name}/Time/Epoch', epoch_time, epoch)
        if strategy_name != 'baseline':
            writer.add_scalar(f'{strategy_name}/Repeats/Avg', avg_repeats, epoch)
        
        print(f"{strategy_name} - Epoch {epoch+1}/{args.epochs}, 训练时间: {epoch_time:.2f}秒")
        print(f"  训练损失: {avg_loss:.4f}, 困惑度: {avg_perplexity:.4f}")
        print(f"  验证损失: {val_loss:.4f}, 困惑度: {val_perplexity:.4f}")
        print(f"  测试损失: {test_loss:.4f}, 困惑度: {test_perplexity:.4f}")
        if strategy_name != 'baseline':
            print(f"  平均重复次数: {avg_repeats:.2f}")

        # 记录历史（便于落盘/后处理）
        strategy.metrics_history['epoch'].append(epoch)
        strategy.metrics_history['train_loss'].append(float(avg_loss))
        strategy.metrics_history['val_loss'].append(float(val_loss))
        strategy.metrics_history['test_loss'].append(float(test_loss))
        strategy.metrics_history['train_ppl'].append(float(avg_perplexity))
        strategy.metrics_history['val_ppl'].append(float(val_perplexity))
        strategy.metrics_history['test_ppl'].append(float(test_perplexity))
        strategy.metrics_history['avg_repeats'].append(float(avg_repeats))
        # 累计 outlier 数
        outlier_cum = len(getattr(strategy, 'outlier_records', [])) if hasattr(strategy, 'outlier_records') else 0
        strategy.metrics_history['outliers_cum'].append(int(outlier_cum))
    
    total_time = time.time() - start_time
    print(f"{strategy_name} 策略训练完成! 总时间: {total_time:.2f}秒")
    
    # 保存指标历史
    save_path = os.path.join(args.save_dir, f'{strategy_name}_metrics.npy')
    np.save(save_path, strategy.metrics_history)

    # 保存 outlier 记录（便于人工复核）
    try:
        if hasattr(strategy, 'outlier_records') and len(getattr(strategy, 'outlier_records', [])) > 0:
            import json
            out_path = os.path.join(args.save_dir, f'{strategy_name}_outliers.jsonl')
            with open(out_path, 'w', encoding='utf-8') as f:
                for rec in strategy.outlier_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"保存 outlier 记录失败: {e}")
    
    return strategy


def train(args):
    # 规范化结果保存目录：相对路径一律相对于本脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(script_dir, args.save_dir)

    # 创建结果保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建唯一的TensorBoard日志目录
    tensorboard_base_dir = os.path.join(args.save_dir, 'tensorboard')
    tensorboard_dir = os.path.join(tensorboard_base_dir, f"{args.dataset}_{int(time.time())}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 清理当前实验的TensorBoard日志（如果存在）
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # 加载数据
    data_loader = TextDataLoader(dataset_name=args.dataset, tokenizer_name=args.model_name, 
                                batch_size=args.batch_size, max_length=args.max_length)
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # 训练所有指定的策略
    strategies = {}
    for strategy_name in args.strategies:
        strategies[strategy_name] = train_single_strategy(strategy_name, args, data_loader, writer)
    
    # 关闭TensorBoard writer
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)