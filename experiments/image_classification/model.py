import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from collections import deque
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

# --- [模型、数据、测试函数部分保持不变] ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_stack = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_stack = nn.Sequential(nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        x = self.conv_stack(x); x = x.view(x.size(0), -1); x = self.fc_stack(x); return x

def get_data_loaders(batch_size, shuffle):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle); test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False); return train_loader, test_loader

def test(model, device, test_loader, criterion):
    model.eval(); test_loss = 0; correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device); output = model(data); test_loss += criterion(output, target).item(); pred = output.argmax(dim=1, keepdim=True); correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader); accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss

# --- 重新设计的调度网络 ---
class SchedulerMLP_Binary(nn.Module):
    """
    新的调度网络，输出2个动作的概率：
    动作0: 不额外重复 (总共训练1次)
    动作1: 额外重复1次 (总共训练2次)
    """
    def __init__(self, input_size, hidden_size=16):
        super(SchedulerMLP_Binary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2), # 输出层大小为2
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.network(x)


# --- [主执行逻辑] ---
if __name__ == '__main__':
    # --- 1. 全局超参数设置 ---
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    SCHEDULER_LR = 1e-4
    CONTEXT_WINDOW_SIZE = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # --- 2. 统一的数据加载器 ---
    train_loader, test_loader = get_data_loaders(BATCH_SIZE, shuffle=True)
    
    # --- 3. 创建所有独立的实验环境 ---
    # a. 基线
    model_base = SimpleCNN().to(device); optimizer_base = optim.SGD(model_base.parameters(), lr=LEARNING_RATE)
    # b. 可学习ABR (Loss only)
    main_model_v1 = SimpleCNN().to(device); main_model_v1.load_state_dict(model_base.state_dict()); main_optimizer_v1 = optim.SGD(main_model_v1.parameters(), lr=LEARNING_RATE)
    scheduler_model_v1 = SchedulerMLP_Binary(input_size=1).to(device); scheduler_optimizer_v1 = optim.Adam(scheduler_model_v1.parameters(), lr=SCHEDULER_LR)
    # c. 可学习ABR (with Context)
    main_model_v2 = SimpleCNN().to(device); main_model_v2.load_state_dict(model_base.state_dict()); main_optimizer_v2 = optim.SGD(main_model_v2.parameters(), lr=LEARNING_RATE)
    scheduler_model_v2 = SchedulerMLP_Binary(input_size=2).to(device); scheduler_optimizer_v2 = optim.Adam(scheduler_model_v2.parameters(), lr=SCHEDULER_LR)

    # --- 4. 统一的实验结果存储 ---
    results = {
        "Baseline": {"accuracies": [], "losses": []},
        "Learnable ABR (Loss only)": {"accuracies": [], "losses": []},
        "Learnable ABR (with Context)": {"accuracies": [], "losses": []}
    }

    # --- 5. 全新的“同步训练”循环 ---
    for epoch in range(1, EPOCHS + 1):
        print("\n" + "="*60 + f"\n========== Starting Epoch {epoch}/{EPOCHS} ==========\n" + "="*60)

        log_probs_epoch = { "v1": [], "v2": [] }
        recent_losses_v2 = deque(maxlen=CONTEXT_WINDOW_SIZE)
        
        for m in [model_base, main_model_v1, scheduler_model_v1, main_model_v2, scheduler_model_v2]: m.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            # --- a. 基线训练 ---
            optimizer_base.zero_grad(); output_base = model_base(data); loss_base = criterion(output_base, target); loss_base.backward(); optimizer_base.step()

            # --- b. 可学习ABR (Loss only) ---
            # 步骤1: 先正常训练一次
            main_optimizer_v1.zero_grad(); output_v1 = main_model_v1(data); loss_v1 = criterion(output_v1, target); loss_v1.backward(); main_optimizer_v1.step()
            # 步骤2: 根据这次的loss决策是否“补课”
            scheduler_input_v1 = torch.tensor([loss_v1.item()], dtype=torch.float32).to(device)
            action_probs_v1 = scheduler_model_v1(scheduler_input_v1)
            m_v1 = Categorical(action_probs_v1); action_v1 = m_v1.sample(); log_probs_epoch["v1"].append(m_v1.log_prob(action_v1))
            if action_v1.item() == 1: # 如果决策是“重复”
                main_optimizer_v1.zero_grad(); output_v1 = main_model_v1(data); loss_v1 = criterion(output_v1, target); loss_v1.backward(); main_optimizer_v1.step()
            
            # --- c. 可学习ABR (with Context) ---
            # 步骤1: 先正常训练一次
            main_optimizer_v2.zero_grad(); output_v2 = main_model_v2(data); loss_v2 = criterion(output_v2, target); loss_v2.backward(); main_optimizer_v2.step()
            current_loss_v2 = loss_v2.item()
            # 步骤2: 根据这次的loss和上下文决策是否“补课”
            avg_loss_v2 = sum(recent_losses_v2) / len(recent_losses_v2) if recent_losses_v2 else 0.0
            scheduler_input_v2 = torch.tensor([current_loss_v2, avg_loss_v2], dtype=torch.float32).to(device)
            action_probs_v2 = scheduler_model_v2(scheduler_input_v2)
            m_v2 = Categorical(action_probs_v2); action_v2 = m_v2.sample(); log_probs_epoch["v2"].append(m_v2.log_prob(action_v2))
            if action_v2.item() == 1: # 如果决策是“重复”
                main_optimizer_v2.zero_grad(); output_v2 = main_model_v2(data); loss_v2 = criterion(output_v2, target); loss_v2.backward(); main_optimizer_v2.step()
            if current_loss_v2 > 0: recent_losses_v2.append(current_loss_v2)

        # --- Epoch结束后，评估所有模型并更新调度网络 ---
        acc_base, loss_base = test(model_base, device, test_loader, criterion); results["Baseline"]["accuracies"].append(acc_base); results["Baseline"]["losses"].append(loss_base)
        acc_v1, loss_v1 = test(main_model_v1, device, test_loader, criterion); results["Learnable ABR (Loss only)"]["accuracies"].append(acc_v1); results["Learnable ABR (Loss only)"]["losses"].append(loss_v1)
        acc_v2, loss_v2 = test(main_model_v2, device, test_loader, criterion); results["Learnable ABR (with Context)"]["accuracies"].append(acc_v2); results["Learnable ABR (with Context)"]["losses"].append(loss_v2)
        
        print(f"--- Epoch {epoch} Results ---")
        print(f"Baseline                      | Acc: {acc_v1:.2f}% | Loss: {loss_v1:.4f}")
        print(f"Learnable ABR (Loss only)     | Acc: {acc_v2:.2f}% | Loss: {loss_v2:.4f}")
        print(f"Learnable ABR (with Context)  | Acc: {acc_base:.2f}% | Loss: {loss_base:.4f}")

        # 更新调度网络
        prev_acc_v1 = results["Learnable ABR (Loss only)"]["accuracies"][-2] if epoch > 1 else 8.0; reward_v1 = acc_v1 - prev_acc_v1
        scheduler_optimizer_v1.zero_grad(); policy_loss_v1 = torch.stack([-log_prob * reward_v1 for log_prob in log_probs_epoch["v1"]]).sum(); policy_loss_v1.backward(); scheduler_optimizer_v1.step()
        print(f"Scheduler (Loss only) Updated with Reward: {reward_v1:.4f}")
        
        prev_acc_v2 = results["Learnable ABR (with Context)"]["accuracies"][-2] if epoch > 1 else 8.0; reward_v2 = acc_v2 - prev_acc_v2
        scheduler_optimizer_v2.zero_grad(); policy_loss_v2 = torch.stack([-log_prob * reward_v2 for log_prob in log_probs_epoch["v2"]]).sum(); policy_loss_v2.backward(); scheduler_optimizer_v2.step()
        print(f"Scheduler (with Context) Updated with Reward: {reward_v2:.4f}")

    # --- 6. 最终结果可视化 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14)); fig.suptitle('Synchronized Comparison of Training Strategies (v2)', fontsize=16)
    markers = ['s', 'o', '^']; styles = ['--', '-', ':']; colors = ['gray', 'blue', 'green']
    
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(range(1, EPOCHS + 1), data['accuracies'], marker=markers[i], linestyle=styles[i], color=colors[i], label=name)
        ax2.plot(range(1, EPOCHS + 1), data['losses'], marker=markers[i], linestyle=styles[i], color=colors[i], label=name)

    ax1.set_title('Test Accuracy vs. Epochs'); ax1.set_ylabel('Test Accuracy (%)'); ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend(); ax1.set_xticks(range(1, EPOCHS + 1))
    ax2.set_title('Test Loss vs. Epochs'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Average Test Loss'); ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend(); ax2.set_xticks(range(1, EPOCHS + 1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
