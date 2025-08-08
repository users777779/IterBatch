import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import time
import shutil
from model import LLaMAWithLoRA, SchedulerLLaMA
from data_loader import TextDataLoader
from strategy import BaselineStrategy, ABRStrategy, LearnableSchedulingStrategy, SlidingWindowStrategy
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
                        choices=['baseline', 'abr', 'learnable', 'window'],
                        help='训练策略选择 (baseline/abr/learnable/window)')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA矩阵的秩')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA缩放因子')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA层的dropout率')
    parser.add_argument('--loss_threshold', type=float, default=1.0, help='ABR策略的损失阈值')
    parser.add_argument('--max_repeats', type=int, default=5, help='ABR策略的最大重复次数')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口策略的窗口大小')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--max_length', type=int, default=512, help='序列最大长度')
    parser.add_argument('--primer_batches', type=int, default=0, help='预热训练的批次数（使用约1000条高质量样本作为“药引子”）')
    # 新增：快速试跑控制
    parser.add_argument('--max_train_batches', type=int, default=50, help='单个epoch最多训练批次数（用于快速试跑，-1表不限制）')
    parser.add_argument('--max_eval_batches', type=int, default=50, help='验证/测试最多评估批次数（用于快速试跑，-1表不限制）')
    # 新增：步级日志记录间隔（单位：步）；设置为1表示每步记录
    parser.add_argument('--log_every_n_steps', type=int, default=1, help='每多少步记录一次训练指标到TensorBoard')
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
        return SlidingWindowStrategy(model, criterion, optimizer, args.window_size, args.loss_threshold, args.max_repeats)
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
        epoch_start_time = time.time()
        total_loss = 0
        total_perplexity = 0
        total_repeats = 0
        count = 0
        
        # 训练一个epoch（限制最多批次数）
        batch_seen = 0
        for batch in train_loader:
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
            
            total_loss += loss * batch['input_ids'].size(0)
            total_perplexity += perplexity * batch['input_ids'].size(0)
            total_repeats += repeats
            count += batch['input_ids'].size(0)
            batch_seen += 1
            if args.max_train_batches is not None and args.max_train_batches > 0 and batch_seen >= args.max_train_batches:
                break
        
        avg_loss = total_loss / count
        avg_perplexity = total_perplexity / count
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
    
    total_time = time.time() - start_time
    print(f"{strategy_name} 策略训练完成! 总时间: {total_time:.2f}秒")
    
    # 保存指标历史
    save_path = os.path.join(args.save_dir, f'{strategy_name}_metrics.npy')
    np.save(save_path, strategy.metrics_history)
    
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