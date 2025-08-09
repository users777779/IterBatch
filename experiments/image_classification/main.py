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
import os

# ---- Model Definition ----
# DeepCNN：用于CIFAR10图像分类的深度卷积神经网络
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()  # 继承nn.Module的初始化
        # 卷积层序列 - 更深的网络结构
        self.conv = nn.Sequential(
            # 第一个卷积块：输入3通道(RGB)，输出32通道
            nn.Conv2d(3, 32, 3, 1, 1),     # 输入通道3(RGB)，输出32通道，3x3卷积核，步长1，填充1
            nn.BatchNorm2d(32),             # 批归一化，提高训练稳定性
            nn.ReLU(),                      # 激活函数
            nn.Conv2d(32, 32, 3, 1, 1),    # 第二个卷积层：32->32通道
            nn.BatchNorm2d(32),             # 批归一化
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(2, 2),             # 2x2最大池化，步长2，图像尺寸减半
            
            # 第二个卷积块：32->64通道
            nn.Conv2d(32, 64, 3, 1, 1),    # 32通道输入，64通道输出
            nn.BatchNorm2d(64),             # 批归一化
            nn.ReLU(),                      # 激活函数
            nn.Conv2d(64, 64, 3, 1, 1),    # 第二个卷积层：64->64通道
            nn.BatchNorm2d(64),             # 批归一化
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(2, 2),             # 2x2最大池化，步长2，图像尺寸再次减半
            
            # 第三个卷积块：64->128通道
            nn.Conv2d(64, 128, 3, 1, 1),   # 64通道输入，128通道输出
            nn.BatchNorm2d(128),            # 批归一化
            nn.ReLU(),                      # 激活函数
            nn.Conv2d(128, 128, 3, 1, 1),  # 第二个卷积层：128->128通道
            nn.BatchNorm2d(128),            # 批归一化
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(2, 2),             # 2x2最大池化，步长2，图像尺寸再次减半
            
            # Dropout层：防止过拟合
            nn.Dropout2d(0.25)              # 25%的dropout率
        )
        # 全连接层序列
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),    # 展平后的特征图(128通道*4*4)连接到512个神经元
            nn.ReLU(),                      # 激活函数
            nn.Dropout(0.5),                # 50%的dropout率
            nn.Linear(512, 256),            # 512->256个神经元
            nn.ReLU(),                      # 激活函数
            nn.Dropout(0.5),                # 50%的dropout率
            nn.Linear(256, 10)              # 最后输出层，10个类别(CIFAR10的10个类别)
        )
    
    def forward(self, x):                  # 前向传播函数
        x = self.conv(x)                   # 通过卷积层序列
        x = x.view(x.size(0), -1)          # 展平特征图：[batch, 128, 4, 4] -> [batch, 128*4*4]
        x = self.fc(x)                     # 通过全连接层序列
        return x                           # 返回logits，形状[batch, 10]

# DecisionMLP：决策网络，用于判断是否需要重复训练当前batch
class DecisionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):  # 增加隐藏层维度以适应更复杂的决策
        super().__init__()                 # 继承nn.Module的初始化
        # 全连接层序列
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入层：input_dim维(loss值或loss+context) -> hidden_dim维
            nn.ReLU(),                         # 激活函数
            nn.Dropout(0.2),                   # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim//2),  # 第二个隐藏层
            nn.ReLU(),                         # 激活函数
            nn.Dropout(0.2),                   # 添加dropout
            nn.Linear(hidden_dim//2, 2)        # 输出层：hidden_dim//2维 -> 2维(表示两个动作的logits)
        )                                      #logits是最后一层的原始输出
    
    def forward(self, x):                  # 前向传播函数 x是loss值或loss+context
        return self.net(x)                 # 返回logits，形状[batch, 2]，两个动作的logits

def get_loaders(batch_size):
    # 定义数据预处理变换 - CIFAR10专用
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，增加数据增强
        transforms.RandomHorizontalFlip(),      # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10标准化参数
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10标准化参数
    ])
    
    train_set = datasets.CIFAR10('experiments/image_classification/data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10('experiments/image_classification/data', train=False, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data) # 模型输出： out 形状: [batch_size, 10]（10个类别的logits）
            loss = criterion(out, target)#指导梯度下降，告诉模型如何调参
            loss_sum += loss.item()
            pred = out.argmax(dim=1)
            # 取最大概率的类别作为预测结果
            # argmax(dim=1): 在类别维度上取最大值索引
            # pred 形状: [batch_size]，每个元素是预测的类别索引（0-9）
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total, loss_sum / len(loader)

def main():
    # ---- Hyperparameters ----
        # ---- 超参数设置 ----
    epochs = 50                            # 训练轮数：模型将遍历整个训练集50次
    batch_size = 64                        # 批大小：每次处理64个样本（CIFAR10较大，适当减小batch_size）
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
    import shutil
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
    model_sup_v1 = DeepCNN().to(device)   # 创建主分类模型
    model_sup_v1.load_state_dict(base_init) # 加载相同的初始权重（确保公平比较）
    opt_sup_v1 = optim.Adam(model_sup_v1.parameters(), lr=lr)  # 主模型优化器
    decider_sup_v1 = DecisionMLP(1).to(device)  # 决策网络：输入维度1（仅当前loss）
    opt_decider_sup_v1 = optim.Adam(decider_sup_v1.parameters(), lr=scheduler_lr)  # 决策网络优化器（用Adam）
    
    # === 监督学习 context ABR实验 ===
    model_sup_v2 = DeepCNN().to(device)   # 创建主分类模型
    model_sup_v2.load_state_dict(base_init) # 加载相同的初始权重
    opt_sup_v2 = optim.Adam(model_sup_v2.parameters(), lr=lr)  # 主模型优化器
    decider_sup_v2 = DecisionMLP(2).to(device)  # 决策网络：输入维度2（当前loss + 历史均值）
    opt_decider_sup_v2 = optim.Adam(decider_sup_v2.parameters(), lr=scheduler_lr)  # 决策网络优化器

    # ---- TensorBoard日志记录器 ----
    writer = SummaryWriter(log_dir='experiments/image_classification/result/runs/iterbatch_exp')  # 创建TensorBoard写入器

        # ---- 训练循环开始 ----
    criterion = nn.CrossEntropyLoss()        # 主模型的损失函数（分类任务标准损失）
    criterion_sup = nn.CrossEntropyLoss()    # 监督学习决策网络的损失函数
    
    # ---- 初始化记录列表（用于存储每个epoch的结果） ----
    accs_base, losses_base = [], []         
    accs_sup_v1, losses_sup_v1 = [], []    
    accs_sup_v2, losses_sup_v2 = [], []     
    
    # ---- 开始训练循环 ----
    for epoch in range(1, epochs+1):        # 遍历每个epoch（1到15）
        # 将所有模型设置为训练模式（启用dropout、batch norm等）
        model_base.train(); model_sup_v1.train(); model_sup_v2.train()
        decider_sup_v1.train(); decider_sup_v2.train()
        
        # ---- 初始化滑动窗口（用于context决策网络） ----
        recent_losses_sup_v2 = deque(maxlen=window_size)  # 监督学习context的历史loss队列
        
        # ---- 初始化每个epoch的指标统计 ----
        loss_sum_base, correct_base, total_base = 0, 0, 0           
        loss_sum_sup_v1, correct_sup_v1, total_sup_v1 = 0, 0, 0    
        loss_sum_sup_v2, correct_sup_v2, total_sup_v2 = 0, 0, 0     
        
        # ---- 遍历每个batch ----
        for data, target in train_loader:
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
            opt_sup_v1.zero_grad()
            out_sup_v1 = model_sup_v1(data)
            loss_sup_v1 = criterion(out_sup_v1, target)
            loss_sup_v1.backward()
            opt_sup_v1.step()
            loss_sum_sup_v1 += loss_sup_v1.item()
            pred_sup_v1 = out_sup_v1.argmax(dim=1)
            correct_sup_v1 += (pred_sup_v1 == target).sum().item()
            total_sup_v1 += target.size(0)
            # 决策网络监督学习（第一次）
            decider_input_sup_v1 = torch.tensor([[loss_sup_v1.item()]], dtype=torch.float32, device=device)
            action_logits_sup_v1 = decider_sup_v1(decider_input_sup_v1)
            action_probs_sup_v1 = torch.softmax(action_logits_sup_v1, dim=-1)
            
            # 根据概率分布随机采样决策
            m_sup_v1 = torch.distributions.Categorical(action_probs_sup_v1)
            action_sup_v1 = m_sup_v1.sample()
            
            # 监督学习标签：使用采样的动作作为标签
            label_sup_v1 = action_sup_v1
            loss_decider_sup_v1 = criterion_sup(action_logits_sup_v1, label_sup_v1)
            opt_decider_sup_v1.zero_grad()
            loss_decider_sup_v1.backward()
            opt_decider_sup_v1.step()
            
            # 如果决策是重复训练
            if action_sup_v1.item() == 1:
                # 重复训练主模型（不更新决策网络，不判断动作）
                opt_sup_v1.zero_grad()
                out_sup_v1 = model_sup_v1(data)
                loss_sup_v1 = criterion(out_sup_v1, target)
                loss_sup_v1.backward()
                opt_sup_v1.step()
                loss_sum_sup_v1 += loss_sup_v1.item()
                pred_sup_v1 = out_sup_v1.argmax(dim=1)
                correct_sup_v1 += (pred_sup_v1 == target).sum().item()
                total_sup_v1 += target.size(0)
            # --- 监督学习 context ---
            opt_sup_v2.zero_grad()
            out_sup_v2 = model_sup_v2(data)
            loss_sup_v2 = criterion(out_sup_v2, target)
            loss_sup_v2.backward()
            opt_sup_v2.step()
            loss_sum_sup_v2 += loss_sup_v2.item()
            pred_sup_v2 = out_sup_v2.argmax(dim=1)
            correct_sup_v2 += (pred_sup_v2 == target).sum().item()
            total_sup_v2 += target.size(0)
            avg_loss_sup_v2 = np.mean(recent_losses_sup_v2) if recent_losses_sup_v2 else 0.0
            decider_input_sup_v2 = torch.tensor([[loss_sup_v2.item(), avg_loss_sup_v2]], dtype=torch.float32, device=device)
            action_logits_sup_v2 = decider_sup_v2(decider_input_sup_v2)
            action_probs_sup_v2 = torch.softmax(action_logits_sup_v2, dim=-1)
            
            # 根据概率分布随机采样决策
            m_sup_v2 = torch.distributions.Categorical(action_probs_sup_v2)
            action_sup_v2 = m_sup_v2.sample()
            
            # 监督学习标签：使用采样的动作作为标签
            label_sup_v2 = action_sup_v2
            loss_decider_sup_v2 = criterion_sup(action_logits_sup_v2, label_sup_v2)
            opt_decider_sup_v2.zero_grad()
            loss_decider_sup_v2.backward()
            opt_decider_sup_v2.step()
            recent_losses_sup_v2.append(loss_sup_v2.item())
            
            # 如果决策是重复训练
            if action_sup_v2.item() == 1:
                # 重复训练主模型（不更新决策网络，不判断动作）
                opt_sup_v2.zero_grad()
                out_sup_v2 = model_sup_v2(data)
                loss_sup_v2 = criterion(out_sup_v2, target)
                loss_sup_v2.backward()
                opt_sup_v2.step()
                loss_sum_sup_v2 += loss_sup_v2.item()
                pred_sup_v2 = out_sup_v2.argmax(dim=1)
                correct_sup_v2 += (pred_sup_v2 == target).sum().item()
                total_sup_v2 += target.size(0)
                recent_losses_sup_v2.append(loss_sup_v2.item())
        
        # ---- Epoch结束：评估所有模型 ----
        acc_base, test_loss_base = evaluate(model_base, test_loader, device)      # 评估基线模型
        acc_sup_v1, test_loss_sup_v1 = evaluate(model_sup_v1, test_loader, device)  # 评估监督学习loss-only
        acc_sup_v2, test_loss_sup_v2 = evaluate(model_sup_v2, test_loader, device)  # 评估监督学习context
        
        # ---- 记录每个epoch的结果 ----
        accs_base.append(acc_base); losses_base.append(test_loss_base)           # 记录基线结果
        accs_sup_v1.append(acc_sup_v1); losses_sup_v1.append(test_loss_sup_v1) # 记录监督学习loss-only结果
        accs_sup_v2.append(acc_sup_v2); losses_sup_v2.append(test_loss_sup_v2) # 记录监督学习context结果
        
        # ---- 日志输出 ----
        print(f"Epoch {epoch}: Baseline acc={acc_base:.4f} loss={test_loss_base:.4f} | SupLossNet acc={acc_sup_v1:.4f} loss={test_loss_sup_v1:.4f} | SupCtxNet acc={acc_sup_v2:.4f} loss={test_loss_sup_v2:.4f}")
        # TensorBoard 分别记录每个实验的指标
        # Baseline实验
        writer.add_scalar('Baseline/Accuracy', acc_base, epoch)
        writer.add_scalar('Baseline/Loss', test_loss_base, epoch)
        # 监督学习 loss-only ABR实验
        writer.add_scalar('SupLossNet/Accuracy', acc_sup_v1, epoch)
        writer.add_scalar('SupLossNet/Loss', test_loss_sup_v1, epoch)
        # 监督学习 context ABR实验
        writer.add_scalar('SupCtxNet/Accuracy', acc_sup_v2, epoch)
        writer.add_scalar('SupCtxNet/Loss', test_loss_sup_v2, epoch)
    writer.close()
    print('Training complete. Use TensorBoard to view results.')
    # ---- Matplotlib 可视化 ----
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accs_base, 'o-', label='Baseline')
    plt.plot(epochs_range, accs_sup_v1, 's-', label='SupLossNet')
    plt.plot(epochs_range, accs_sup_v2, '^-', label='SupCtxNet')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CIFAR10 Test Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, losses_base, 'o-', label='Baseline')
    plt.plot(epochs_range, losses_sup_v1, 's-', label='SupLossNet')
    plt.plot(epochs_range, losses_sup_v2, '^-', label='SupCtxNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CIFAR10 Test Loss vs. Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('experiments/image_classification/result/iterbatch_exp_results.png')
    plt.show()

if __name__ == '__main__':
    main()
