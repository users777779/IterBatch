"""
更新日志

2025-08-11
- 新增 generate_supervision_signal_v1：基于近期历史（recent_losses / recent_accs）的动态阈值标签生成。
- v1 决策网络输入对齐为 4 维：[loss, acc, loss_ratio, acc_gap]。
- v2 决策网络输入对齐为 6 维：[loss, avg_loss, acc, avg_acc, loss_ratio, acc_gap]。
- 决策策略：前 10 个 epoch 采用 ε-greedy 探索（线性衰减至 0），之后使用 argmax 决策。
- 重复训练步学习率缩放为 0.8x，并限制每个 epoch 最大重复比例为 30%。
- 新增滑动窗口缓存 recent_losses_* / recent_accs_*，用于构造动态特征与监督信号。
- 放宽触发阈值：loss阈值从1.5/1.3降为1.2/1.1，acc阈值从0.7/0.9放宽到0.9/0.95。
- 增加误分类比例门限：acc < 0.7 才考虑重复，减少在已较好batch上浪费预算。
- 添加关键统计记录：每epoch重复步次数/比例、label=1比例、action=1比例，便于诊断效果。

2025-08-12 (智能重复训练版本)
- 智能重复训练：最多重复3次，使用递增学习率(1.5x, 2.0x, 2.5x)。
- 效果验证：每次重复后检查loss/acc是否改善，改善即停止。
- 详细统计：记录改善率、平均重复步数等关键指标。
- 目标：提高重复训练的有效性，确保每次重复都有实际改善。

2025-08-12 (保守优化版本)
- 降低学习率：从1.5x-2.5x改为1.2x-1.6x，避免过度训练。
- 严格改善检查：要求loss下降>1%或acc提升>0.5%才算改善。
- 目标：解决重复训练效果不如基线的问题，提高训练稳定性。

2025-08-12 (重复训练优化版本)
- 智能重复训练：最多重复3次，使用递增学习率(1.5x, 2.0x, 2.5x)。
- 效果验证：每次重复后检查loss/acc是否改善，改善即停止。
- 详细统计：记录改善率、平均重复步数等关键指标。
- 目标：提高重复训练的有效性，确保每次重复都有实际改善。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import shutil
import os

def generate_supervision_signal(
    current_loss: float,
    current_acc: float,
    recent_losses,
    recent_accs,
    device
):
    """
    V1: 使用动态阈值和近期历史表现来生成监督信号。
    需要传入 recent_losses 与 recent_accs（例如长度为 window_size 的 deque）。
    """
    # 如果历史记录不足，默认不重复，以稳定早期训练（降低门槛从10到5）
    if len(recent_losses) < 5 or len(recent_accs) < 5:
        return torch.tensor([0], dtype=torch.long, device=device)

    # 1. 计算动态基准
    avg_loss = sum(recent_losses) / len(recent_losses)
    avg_acc = sum(recent_accs) / len(recent_accs)

    should_repeat = False

    # 2. 设计动态规则（大幅放宽阈值，提高触发率）
    loss_threshold_1 = 1.02
    loss_threshold_2 = 1.01
    acc_threshold_1 = 0.995
    acc_threshold_2 = 0.998
    
    # 规则A: 灾难性表现 (Loss远高于平均值 或 Acc远低于平均值)
    if current_loss > avg_loss * loss_threshold_1 or current_acc < avg_acc * acc_threshold_1:
        should_repeat = True
    # 规则B: 表现不佳 (Loss较高 且 Acc较低)
    elif current_loss > avg_loss * loss_threshold_2 and current_acc < avg_acc * acc_threshold_2:
        should_repeat = True
    # 规则C: 退步惩罚 (Loss高于平均 且 Acc也低于平均)
    elif current_loss > avg_loss and current_acc < avg_acc:
        should_repeat = True
    # 规则D: 激进策略 - 只要不如历史平均就重复（大幅提高触发率）
    elif current_loss > avg_loss or current_acc < avg_acc:
        should_repeat = True
    
    # 3. 使用相对门槛替代绝对门槛：放宽相对下降要求
    acc_gap = current_acc - avg_acc
    gap_threshold = -0.005
    if should_repeat and acc_gap >= gap_threshold:
        should_repeat = False

    # 4. 转换为标签
    label = 1 if should_repeat else 0
    return torch.tensor([label], dtype=torch.long, device=device)



# ---- Model Definition ----
# DeepCNN：用于CIFAR10图像分类的深度卷积神经网络
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层序列 - 更深的网络结构
        self.conv = nn.Sequential(
            # 第一个卷积块：输入3通道(RGB)，输出32通道
            nn.Conv2d(3, 32, 3, 1, 1),     
            nn.BatchNorm2d(32),             
            nn.ReLU(),                      
            nn.Conv2d(32, 32, 3, 1, 1),    
            nn.BatchNorm2d(32),             
            nn.ReLU(),                      
            nn.MaxPool2d(2, 2),            
            
            # 第二个卷积块：32->64通道
            nn.Conv2d(32, 64, 3, 1, 1),   
            nn.BatchNorm2d(64),             
            nn.ReLU(),                      
            nn.Conv2d(64, 64, 3, 1, 1),    
            nn.BatchNorm2d(64),             
            nn.ReLU(),                      
            nn.MaxPool2d(2, 2),             
            
            # 第三个卷积块：64->128通道
            nn.Conv2d(64, 128, 3, 1, 1),   
            nn.BatchNorm2d(128),            
            nn.ReLU(),                      
            nn.Conv2d(128, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),            
            nn.ReLU(),                      
            nn.MaxPool2d(2, 2),             
            
            # Dropout层：防止过拟合
            nn.Dropout2d(0.25)              
        )
        # 全连接层序列
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),    
            nn.ReLU(),                      
            nn.Dropout(0.5),               
            nn.Linear(512, 256),            
            nn.ReLU(),                      
            nn.Dropout(0.5),                
            nn.Linear(256, 10)              
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# DecisionMLP：决策网络，用于判断是否需要重复训练当前batch
class DecisionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # 改进的决策网络架构
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)

def get_loaders(batch_size):
    # 定义数据预处理变换 - CIFAR10专用
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),      
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  
    ])
    
    train_set = datasets.CIFAR10('experiments/image_classification/data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10('experiments/image_classification/data', train=False, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)
            loss_sum += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total, loss_sum / len(loader)

def main():
    # ---- Hyperparameters ----
        # ---- 超参数设置 ----
    epochs = 50                            # 训练轮数：模型将遍历整个训练集80次
    batch_size = 32                        # 批大小：每次处理64个样本（CIFAR10较大，适当减小batch_size）
    lr = 0.001                              # 主模型学习率：Adam优化器的步长（更深的网络使用更小的学习率）
    scheduler_lr = 1e-4                     # 决策网络学习率：Adam优化器的步长（比主模型小10倍）
    window_size = 10                        # 滑动窗口大小：Context决策网络使用前10个batch的loss均值
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择：有GPU用GPU，否则用CPU
    
    # ---- 随机种子设置（确保实验可重现） ----
    seed = 42                               # 固定随机种子
    torch.manual_seed(seed)                 
    np.random.seed(seed)                    
    random.seed(seed)                       
    
    # ---- 创建输出目录 ----
    # 清理之前的TensorBoard日志，确保只显示当前实验
    if os.path.exists('experiments/image_classification/result/runs/iterbatch_exp'):
        shutil.rmtree('experiments/image_classification/result/runs/iterbatch_exp')
    os.makedirs('experiments/image_classification/result/runs/iterbatch_exp', exist_ok=True)  # 创建TensorBoard日志目录
    os.makedirs('experiments/image_classification/result', exist_ok=True)                     # 创建结果输出目录
    # exist_ok=True：如果目录已存在不报错

    # ---- 数据加载 ----
    train_loader, test_loader = get_loaders(batch_size)  # 获取训练和测试数据加载器

    # ---- 模型初始化 ----
    base_model = DeepCNN().to(device)     # 创建基础模型并移到指定设备
    base_init = base_model.state_dict()     # 保存初始权重参数（字典格式）
    
    # === Baseline实验：标准训练，不使用决策网络 ===
    model_base = DeepCNN().to(device)     # 创建基线模型
    model_base.load_state_dict(base_init)   # 加载相同的初始权重
    opt_base = optim.Adam(model_base.parameters(), lr=lr)  # 为基线模型创建Adam优化器（更深的网络用Adam）
    
    # === 监督学习 loss-only ABR实验 ===
    model_v1 = DeepCNN().to(device)   # 创建主分类模型
    model_v1.load_state_dict(base_init) # 加载相同的初始权重（确保公平比较）
    opt_v1 = optim.Adam(model_v1.parameters(), lr=lr)  # 主模型优化器
    decider_v1 = DecisionMLP(4).to(device)  # 决策网络：输入维度4（loss, acc, loss_ratio, acc_gap）
    opt_decider_v1 = optim.Adam(decider_v1.parameters(), lr=scheduler_lr)  # 决策网络优化器（用Adam）
    
    # === 监督学习 context ABR实验 ===
    model_v2 = DeepCNN().to(device)   # 创建主分类模型
    model_v2.load_state_dict(base_init) # 加载相同的初始权重
    opt_v2 = optim.Adam(model_v2.parameters(), lr=lr)  # 主模型优化器
    decider_v2 = DecisionMLP(6).to(device)  # 决策网络：输入维度6（loss, avg_loss, acc, avg_acc, loss_ratio, acc_gap）
    opt_decider_v2 = optim.Adam(decider_v2.parameters(), lr=scheduler_lr)  # 决策网络优化器

    # ---- TensorBoard日志记录器 ----
    writer = SummaryWriter(log_dir='experiments/image_classification/result/runs/iterbatch_exp')  # 创建TensorBoard日志写入器

        # ---- 训练循环开始 ----
    criterion = nn.CrossEntropyLoss()        # 主模型的损失函数（分类任务标准损失）
    criterion_sup = nn.CrossEntropyLoss()    # 监督学习决策网络的损失函数
    
    # ---- 初始化记录列表（用于存储每个epoch的结果） ----
    accs_base, losses_base = [], []         
    accs_v1, losses_v1 = [], []    
    accs_v2, losses_v2 = [], []     
    
    # ---- 初始化持久化滑动窗口（用于历史表现，不重置） ----
    recent_losses_v1 = deque(maxlen=window_size)
    recent_accs_v1 = deque(maxlen=window_size)
    recent_losses_v2 = deque(maxlen=window_size)
    recent_accs_v2 = deque(maxlen=window_size)
    
    # ---- 初始化自适应阈值标志 ----

    
    # ---- 开始训练循环 ----
    for epoch in range(1, epochs+1):        # 遍历每个epoch（1到15）
        # 将所有模型设置为训练模式（启用dropout、batch norm等）
        model_base.train(); model_v1.train(); model_v2.train()
        decider_v1.train(); decider_v2.train()
        
        # 重复计数与标志
        repeat_count_v1 = 0
        repeat_count_v2 = 0
        epoch_start_flag = True
        
        # ---- 初始化每个epoch的指标统计 ----
        loss_sum_base, correct_base, total_base = 0, 0, 0           
        loss_sum_v1, correct_v1, total_v1 = 0, 0, 0    
        loss_sum_v2, correct_v2, total_v2 = 0, 0, 0     
        
        # ---- 初始化决策统计（用于诊断） ----
        label_1_count_v1, action_1_count_v1 = 0, 0
        label_1_count_v2, action_1_count_v2 = 0, 0
        total_batches = 0
        
        # ---- 初始化重复训练效果统计 ----
        repeat_improved_v1, repeat_improved_v2 = 0, 0
        total_repeat_steps_v1, total_repeat_steps_v2 = 0, 0
        
        # ---- 遍历每个batch ----
        for data, target in train_loader:
            total_batches += 1
            epoch_start_flag = False
            data, target = data.to(device), target.to(device)
            # --- Baseline ---
            opt_base.zero_grad()
            out_base = model_base(data)
            loss_base = criterion(out_base, target)
            loss_base.backward()
            opt_base.step()
            loss_sum_base += loss_base.item()
            pred_base = out_base.argmax(dim=1)
            correct_base += (pred_base == target).sum().item()
            total_base += target.size(0)
            # --- 监督学习 loss-only ---
            opt_v1.zero_grad()
            out_v1 = model_v1(data)
            loss_v1 = criterion(out_v1, target)
            loss_v1.backward()
            opt_v1.step()
            loss_sum_v1 += loss_v1.item()
            pred_v1 = out_v1.argmax(dim=1)
            correct_v1 += (pred_v1 == target).sum().item()
            total_v1 += target.size(0)
            # 决策网络监督学习（第一次）
            # v1 输入: [loss, acc, loss_ratio, acc_gap]
            loss_ratio_v1 = loss_v1.item() / (sum(recent_losses_v1)/len(recent_losses_v1) if recent_losses_v1 else (loss_v1.item()+1e-8))
            acc_v1 = (pred_v1 == target).sum().item() / target.size(0)
            avg_acc_v1 = (sum(recent_accs_v1)/len(recent_accs_v1)) if recent_accs_v1 else acc_v1
            acc_gap_v1 = acc_v1 - avg_acc_v1
            decider_input_v1 = torch.tensor([[loss_v1.item(), acc_v1, loss_ratio_v1, acc_gap_v1]], dtype=torch.float32, device=device)
            action_logits_v1 = decider_v1(decider_input_v1)
            action_probs_v1 = torch.softmax(action_logits_v1, dim=-1)
        
            action_v1 = action_probs_v1.argmax(dim=-1)
            # 使用新的V1监督信号（需依赖 recent_losses_v1/recent_accs_v1 队列）
            label_v1 = generate_supervision_signal(
                loss_v1.item(),
                acc_v1,
                recent_losses_v1,
                recent_accs_v1,
                device
            )
            # 记录决策统计
            if label_v1.item() == 1:
                label_1_count_v1 += 1
            if action_v1.item() == 1:
                action_1_count_v1 += 1
                
            loss_decider_v1 = criterion_sup(action_logits_v1, label_v1)
            opt_decider_v1.zero_grad()
            loss_decider_v1.backward()
            opt_decider_v1.step()
            # 更新v1历史队列（在生成标签与更新后再追加当前样本）
            recent_losses_v1.append(loss_v1.item())
            recent_accs_v1.append(acc_v1)
            #如果没有重复，batch到此结束
            
            # 每个 epoch 限制最大重复比例50%（大幅提高重复步上限）
            max_repeats_v1 = int(0.50 * len(train_loader))
            # 如果决策是重复训练
            if action_v1.item() == 1 and repeat_count_v1 < max_repeats_v1:
                # 记录重复训练前的性能
                with torch.no_grad():
                    out_v1_before = model_v1(data)
                    loss_v1_before = criterion(out_v1_before, target)
                    pred_v1_before = out_v1_before.argmax(dim=1)
                    acc_v1_before = (pred_v1_before == target).sum().item() / target.size(0)
                
                # 重复训练主模型（多次重复，直到改善或达到最大次数）
                max_repeat_steps = 3  # 最多重复3次
                repeat_steps = 0
                improved = False
                
                for repeat_step in range(max_repeat_steps):
                    # 使用更保守的递增学习率：1.2x, 1.4x, 1.6x
                    lr_multiplier = 1.2 + repeat_step * 0.2
                    for g in opt_v1.param_groups:
                        g['lr'] = lr * lr_multiplier
                    
                    opt_v1.zero_grad()
                    out_v1 = model_v1(data)
                    loss_v1 = criterion(out_v1, target)
                    loss_v1.backward()
                    opt_v1.step()
                    repeat_steps += 1
                    
                    # 检查是否改善
                    with torch.no_grad():
                        out_v1_check = model_v1(data)
                        loss_v1_check = criterion(out_v1_check, target)
                        pred_v1_check = out_v1_check.argmax(dim=1)
                        acc_v1_check = (pred_v1_check == target).sum().item() / target.size(0)
                    
                    # 更严格的改善检查：要求loss显著下降或acc显著提升
                    loss_improvement = loss_v1_before.item() - loss_v1_check.item()
                    acc_improvement = acc_v1_check - acc_v1_before
                    
                    # 要求loss下降超过1%或acc提升超过0.5%
                    if loss_improvement > loss_v1_before.item() * 0.01 or acc_improvement > 0.005:
                        improved = True
                        break
                
                # 恢复学习率
                for g in opt_v1.param_groups:
                    g['lr'] = lr
                repeat_count_v1 += 1
                total_repeat_steps_v1 += repeat_steps
                if improved:
                    repeat_improved_v1 += 1
                # 用重复训练后的结果再次更新决策网络
                # 重新计算重复训练后的loss和accuracy
                with torch.no_grad():
                    out_v1_repeat = model_v1(data)
                    loss_v1_repeat = criterion(out_v1_repeat, target)
                    pred_v1_repeat = out_v1_repeat.argmax(dim=1)
                    acc_v1_repeat = (pred_v1_repeat == target).sum().item() / target.size(0)
                # 训练统计改为以重复后的结果为准，避免重复累计
                prev_correct_v1 = (pred_v1_before == target).sum().item()
                new_correct_v1 = (pred_v1_repeat == target).sum().item()
                loss_sum_v1 = loss_sum_v1 - loss_v1_before.item() + loss_v1_repeat.item()
                correct_v1 += new_correct_v1 - prev_correct_v1
                # total_v1 已在重复前累计一次，不再重复累计
                # 使用重复训练后的结果生成新的监督信号（V1）
                label_v1_repeat = generate_supervision_signal(
                    loss_v1_repeat.item(),
                    acc_v1_repeat,
                    recent_losses_v1,
                    recent_accs_v1,
                    device
                )
                # 重新计算决策网络的输入（使用重复训练后的loss等特征）
                avg_acc_v1_repeat = (sum(recent_accs_v1)/len(recent_accs_v1)) if recent_accs_v1 else acc_v1_repeat
                loss_ratio_v1_repeat = loss_v1_repeat.item() / (sum(recent_losses_v1)/len(recent_losses_v1) if recent_losses_v1 else (loss_v1_repeat.item()+1e-8))
                acc_gap_v1_repeat = acc_v1_repeat - avg_acc_v1_repeat
                decider_input_v1_repeat = torch.tensor([[loss_v1_repeat.item(), acc_v1_repeat, loss_ratio_v1_repeat, acc_gap_v1_repeat]], dtype=torch.float32, device=device)
                action_logits_v1_repeat = decider_v1(decider_input_v1_repeat)
                # 使用新的监督信号更新决策网络
                loss_decider_v1_repeat = criterion_sup(action_logits_v1_repeat, label_v1_repeat)
                opt_decider_v1.zero_grad()
                loss_decider_v1_repeat.backward()
                opt_decider_v1.step()
                # 修正历史窗口：将上一条记录替换为重复后的度量
                if len(recent_losses_v1) > 0:
                    recent_losses_v1[-1] = loss_v1_repeat.item()
                if len(recent_accs_v1) > 0:
                    recent_accs_v1[-1] = acc_v1_repeat
            # --- 监督学习 context ---
            opt_v2.zero_grad()
            out_v2 = model_v2(data)
            loss_v2 = criterion(out_v2, target)
            loss_v2.backward()
            opt_v2.step()
            loss_sum_v2 += loss_v2.item()
            pred_v2 = out_v2.argmax(dim=1)
            correct_v2 += (pred_v2 == target).sum().item()
            total_v2 += target.size(0)
            avg_loss_v2 = np.mean(recent_losses_v2) if recent_losses_v2 else 0.0
            # v2 输入: [loss, avg_loss, acc, avg_acc, loss_ratio, acc_gap]
            acc_v2 = (pred_v2 == target).sum().item() / target.size(0)
            avg_acc_v2 = (sum(recent_accs_v2)/len(recent_accs_v2)) if recent_accs_v2 else acc_v2
            loss_ratio_v2 = loss_v2.item() / (avg_loss_v2 if avg_loss_v2 > 0 else (loss_v2.item()+1e-8))
            acc_gap_v2 = acc_v2 - avg_acc_v2
            decider_input_v2 = torch.tensor([[loss_v2.item(), avg_loss_v2, acc_v2, avg_acc_v2, loss_ratio_v2, acc_gap_v2]], dtype=torch.float32, device=device)
            action_logits_v2 = decider_v2(decider_input_v2)
            action_probs_v2 = torch.softmax(action_logits_v2, dim=-1)
            
            action_v2 = action_probs_v2.argmax(dim=-1)
            
            # 监督学习标签：改为使用动态阈值（同时利用v2的历史队列）
            label_v2 = generate_supervision_signal(
                loss_v2.item(),
                acc_v2,
                recent_losses_v2,
                recent_accs_v2,
                device
            )
            # 记录决策统计
            if label_v2.item() == 1:
                label_1_count_v2 += 1
            if action_v2.item() == 1:
                action_1_count_v2 += 1
                
            loss_decider_v2 = criterion_sup(action_logits_v2, label_v2)
            opt_decider_v2.zero_grad()
            loss_decider_v2.backward()
            opt_decider_v2.step()
            # 更新v2历史队列（在生成标签与更新后再追加当前样本）
            recent_losses_v2.append(loss_v2.item())
            recent_accs_v2.append(acc_v2)
            
            # 每个 epoch 限制最大重复比例50%（大幅提高重复步上限）
            max_repeats_v2 = int(0.50 * len(train_loader))
            # 如果决策是重复训练
            if action_v2.item() == 1 and repeat_count_v2 < max_repeats_v2:
                # 记录重复训练前的性能
                with torch.no_grad():
                    out_v2_before = model_v2(data)
                    loss_v2_before = criterion(out_v2_before, target)
                    pred_v2_before = out_v2_before.argmax(dim=1)
                    acc_v2_before = (pred_v2_before == target).sum().item() / target.size(0)
                
                # 重复训练主模型（多次重复，直到改善或达到最大次数）
                max_repeat_steps = 3  # 最多重复3次
                repeat_steps = 0
                improved = False
                
                for repeat_step in range(max_repeat_steps):
                    # 使用更保守的递增学习率：1.2x, 1.4x, 1.6x
                    lr_multiplier = 1.2 + repeat_step * 0.2
                    for g in opt_v2.param_groups:
                        g['lr'] = lr * lr_multiplier
                    
                    opt_v2.zero_grad()
                    out_v2 = model_v2(data)
                    loss_v2 = criterion(out_v2, target)
                    loss_v2.backward()
                    opt_v2.step()
                    repeat_steps += 1
                    
                    # 检查是否改善
                    with torch.no_grad():
                        out_v2_check = model_v2(data)
                        loss_v2_check = criterion(out_v2_check, target)
                        pred_v2_check = out_v2_check.argmax(dim=1)
                        acc_v2_check = (pred_v2_check == target).sum().item() / target.size(0)
                    
                    # 更严格的改善检查：要求loss显著下降或acc显著提升
                    loss_improvement = loss_v2_before.item() - loss_v2_check.item()
                    acc_improvement = acc_v2_check - acc_v2_before
                    
                    # 要求loss下降超过1%或acc提升超过0.5%
                    if loss_improvement > loss_v2_before.item() * 0.01 or acc_improvement > 0.005:
                        improved = True
                        break
                
                # 恢复学习率
                for g in opt_v2.param_groups:
                    g['lr'] = lr
                repeat_count_v2 += 1
                total_repeat_steps_v2 += repeat_steps
                if improved:
                    repeat_improved_v2 += 1
                
                # 用重复训练后的结果再次更新决策网络
                # 重新计算重复训练后的loss和accuracy
                with torch.no_grad():
                    out_v2_repeat = model_v2(data)
                    loss_v2_repeat = criterion(out_v2_repeat, target)
                    pred_v2_repeat = out_v2_repeat.argmax(dim=1)
                    acc_v2_repeat = (pred_v2_repeat == target).sum().item() / target.size(0)
                # 训练统计改为以重复后的结果为准，避免重复累计
                prev_correct_v2 = (pred_v2_before == target).sum().item()
                new_correct_v2 = (pred_v2_repeat == target).sum().item()
                loss_sum_v2 = loss_sum_v2 - loss_v2_before.item() + loss_v2_repeat.item()
                correct_v2 += new_correct_v2 - prev_correct_v2
                # total_v2 已在重复前累计一次，不再重复累计
                # 使用重复训练后的结果生成新的监督信号（V2）
                label_v2_repeat = generate_supervision_signal(
                    loss_v2_repeat.item(),
                    acc_v2_repeat,
                    recent_losses_v2,
                    recent_accs_v2,
                    device
                )   
                # 重新计算决策网络的输入（使用重复训练后的loss和当前的历史平均loss）
                avg_acc_v2_repeat = (sum(recent_accs_v2)/len(recent_accs_v2)) if recent_accs_v2 else acc_v2_repeat
                loss_ratio_v2_repeat = loss_v2_repeat.item() / (avg_loss_v2 if avg_loss_v2 > 0 else (loss_v2_repeat.item()+1e-8))
                acc_gap_v2_repeat = acc_v2_repeat - avg_acc_v2_repeat
                decider_input_v2_repeat = torch.tensor([[loss_v2_repeat.item(), avg_loss_v2, acc_v2_repeat, avg_acc_v2_repeat, loss_ratio_v2_repeat, acc_gap_v2_repeat]], dtype=torch.float32, device=device)
                action_logits_v2_repeat = decider_v2(decider_input_v2_repeat)
                
                # 使用新的监督信号更新决策网络
                loss_decider_v2_repeat = criterion_sup(action_logits_v2_repeat, label_v2_repeat)
                opt_decider_v2.zero_grad()
                loss_decider_v2_repeat.backward()
                opt_decider_v2.step()
                # 修正历史窗口：将上一条记录替换为重复后的度量
                if len(recent_losses_v2) > 0:
                    recent_losses_v2[-1] = loss_v2_repeat.item()
                if len(recent_accs_v2) > 0:
                    recent_accs_v2[-1] = acc_v2_repeat
        
        # ---- Epoch结束：评估所有模型 ----
        acc_base, test_loss_base = evaluate(model_base, test_loader, device)      # 评估基线模型
        acc_v1, test_loss_v1 = evaluate(model_v1, test_loader, device)  # 评估监督学习loss-only
        acc_v2, test_loss_v2 = evaluate(model_v2, test_loader, device)  # 评估监督学习context
        
        # ---- 记录每个epoch的结果 ----
        accs_base.append(acc_base); losses_base.append(test_loss_base)           # 记录基线结果
        accs_v1.append(acc_v1); losses_v1.append(test_loss_v1) # 记录监督学习loss-only结果
        accs_v2.append(acc_v2); losses_v2.append(test_loss_v2) # 记录监督学习context结果
        
        # ---- 计算并记录关键统计（用于诊断） ----
        label_1_ratio_v1 = label_1_count_v1 / total_batches if total_batches > 0 else 0
        action_1_ratio_v1 = action_1_count_v1 / total_batches if total_batches > 0 else 0
        repeat_ratio_v1 = repeat_count_v1 / len(train_loader) if len(train_loader) > 0 else 0
        
        label_1_ratio_v2 = label_1_count_v2 / total_batches if total_batches > 0 else 0
        action_1_ratio_v2 = action_1_count_v2 / total_batches if total_batches > 0 else 0
        repeat_ratio_v2 = repeat_count_v2 / len(train_loader) if len(train_loader) > 0 else 0
        
        # ---- 计算重复训练效果统计 ----
        repeat_improvement_rate_v1 = repeat_improved_v1 / repeat_count_v1 if repeat_count_v1 > 0 else 0
        repeat_improvement_rate_v2 = repeat_improved_v2 / repeat_count_v2 if repeat_count_v2 > 0 else 0
        avg_repeat_steps_v1 = total_repeat_steps_v1 / repeat_count_v1 if repeat_count_v1 > 0 else 0
        avg_repeat_steps_v2 = total_repeat_steps_v2 / repeat_count_v2 if repeat_count_v2 > 0 else 0
        

        
        # ---- 日志输出 ----
        print(f"Epoch {epoch}: Baseline acc={acc_base:.4f} loss={test_loss_base:.4f} | SupLossNet acc={acc_v1:.4f} loss={test_loss_v1:.4f} | SupCtxNet acc={acc_v2:.4f} loss={test_loss_v2:.4f}")
        print(f"  SupLossNet: label_1={label_1_ratio_v1:.3f}, action_1={action_1_ratio_v1:.3f}, repeat={repeat_ratio_v1:.3f}, improve={repeat_improvement_rate_v1:.3f}, steps={avg_repeat_steps_v1:.1f}")
        print(f"  SupCtxNet:  label_1={label_1_ratio_v2:.3f}, action_1={action_1_ratio_v2:.3f}, repeat={repeat_ratio_v2:.3f}, improve={repeat_improvement_rate_v2:.3f}, steps={avg_repeat_steps_v2:.1f}")
        
        # TensorBoard 记录指标 - 将三条曲线汇总到同一卡片
        writer.add_scalars('Accuracy/Test', {
            'Baseline': acc_base,
            'SupLossNet': acc_v1,
            'SupCtxNet': acc_v2,
        }, epoch)
        
        writer.add_scalars('Loss/Test', {
            'Baseline': test_loss_base,
            'SupLossNet': test_loss_v1,
            'SupCtxNet': test_loss_v2,
        }, epoch)
        
        # 决策网络相关指标
        writer.add_scalar('DecisionNet/SupLossNet/Label_1_Ratio', label_1_ratio_v1, epoch)
        writer.add_scalar('DecisionNet/SupLossNet/Action_1_Ratio', action_1_ratio_v1, epoch)
        writer.add_scalar('DecisionNet/SupLossNet/Repeat_Ratio', repeat_ratio_v1, epoch)
        
        writer.add_scalar('DecisionNet/SupCtxNet/Label_1_Ratio', label_1_ratio_v2, epoch)
        writer.add_scalar('DecisionNet/SupCtxNet/Action_1_Ratio', action_1_ratio_v2, epoch)
        writer.add_scalar('DecisionNet/SupCtxNet/Repeat_Ratio', repeat_ratio_v2, epoch)
        
        # 性能提升指标（相对于Baseline）
        writer.add_scalar('Improvement/Accuracy_Improvement/SupLossNet', acc_v1 - acc_base, epoch)
        writer.add_scalar('Improvement/Accuracy_Improvement/SupCtxNet', acc_v2 - acc_base, epoch)
        writer.add_scalar('Improvement/Loss_Improvement/SupLossNet', test_loss_base - test_loss_v1, epoch)
        writer.add_scalar('Improvement/Loss_Improvement/SupCtxNet', test_loss_base - test_loss_v2, epoch)
        
        # 训练统计信息
        writer.add_scalar('Training/Total_Batches', total_batches, epoch)
        writer.add_scalar('Training/Repeat_Count/SupLossNet', repeat_count_v1, epoch)
        writer.add_scalar('Training/Repeat_Count/SupCtxNet', repeat_count_v2, epoch)
        
        # 决策网络学习效果指标
        writer.add_scalar('DecisionNet/Effectiveness/SupLossNet', action_1_ratio_v1 / (label_1_ratio_v1 + 1e-8), epoch)
        writer.add_scalar('DecisionNet/Effectiveness/SupCtxNet', action_1_ratio_v2 / (label_1_ratio_v2 + 1e-8), epoch)
        

        
        # 重复训练效果统计
        writer.add_scalar('RepeatTraining/SupLossNet/Improvement_Rate', repeat_improvement_rate_v1, epoch)
        writer.add_scalar('RepeatTraining/SupLossNet/Avg_Steps', avg_repeat_steps_v1, epoch)
        writer.add_scalar('RepeatTraining/SupCtxNet/Improvement_Rate', repeat_improvement_rate_v2, epoch)
        writer.add_scalar('RepeatTraining/SupCtxNet/Avg_Steps', avg_repeat_steps_v2, epoch)
    writer.close()
    print('Training complete. Use TensorBoard to view results.')
    # ---- Matplotlib 可视化 ----
    epochs_range = range(1, epochs+1)
    
    # 创建两个独立的图表
    # 1. Accuracy比较图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_range, accs_base, 'o-', label='Baseline', linewidth=2, markersize=6)
    plt.plot(epochs_range, accs_v1, 's-', label='SupLossNet', linewidth=2, markersize=6)
    plt.plot(epochs_range, accs_v2, '^-', label='SupCtxNet', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('CIFAR10 Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('experiments/image_classification/result/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Loss比较图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_range, losses_base, 'o-', label='Baseline', linewidth=2, markersize=6)
    plt.plot(epochs_range, losses_v1, 's-', label='SupLossNet', linewidth=2, markersize=6)
    plt.plot(epochs_range, losses_v2, '^-', label='SupCtxNet', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title('CIFAR10 Test Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('experiments/image_classification/result/loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 综合比较图（保存为原来的文件名）
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accs_base, 'o-', label='Baseline', linewidth=2, markersize=6)
    plt.plot(epochs_range, accs_v1, 's-', label='SupLossNet', linewidth=2, markersize=6)
    plt.plot(epochs_range, accs_v2, '^-', label='SupCtxNet', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Test Accuracy', fontsize=11)
    plt.title('CIFAR10 Test Accuracy vs. Epoch', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, losses_base, 'o-', label='Baseline', linewidth=2, markersize=6)
    plt.plot(epochs_range, losses_v1, 's-', label='SupLossNet', linewidth=2, markersize=6)
    plt.plot(epochs_range, losses_v2, '^-', label='SupCtxNet', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Test Loss', fontsize=11)
    plt.title('CIFAR10 Test Loss vs. Epoch', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig('experiments/image_classification/result/iterbatch_exp_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
