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

# ---- Model Definition ----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---- Decision Network ----
class DecisionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# ---- Data Loader ----
def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 数据集下载和缓存路径已限定在 image_classification/data 下
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False)
    return train_loader, test_loader

# ---- Evaluation ----
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

# ---- Main Experiment ----
def main():
    # ---- Hyperparameters ----
    epochs = 10
    batch_size = 64
    lr = 0.01
    scheduler_lr = 1e-4
    window_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---- Data ----
    train_loader, test_loader = get_loaders(batch_size)

    # ---- Model Initialization ----
    base_model = SimpleCNN().to(device)
    base_init = base_model.state_dict()
    # Baseline
    model_base = SimpleCNN().to(device)
    model_base.load_state_dict(base_init)
    opt_base = optim.SGD(model_base.parameters(), lr=lr)
    # Loss-only decision
    model_v1 = SimpleCNN().to(device)
    model_v1.load_state_dict(base_init)
    opt_v1 = optim.SGD(model_v1.parameters(), lr=lr)
    decider_v1 = DecisionMLP(1).to(device)
    opt_decider_v1 = optim.Adam(decider_v1.parameters(), lr=scheduler_lr)
    # Context decision
    model_v2 = SimpleCNN().to(device)
    model_v2.load_state_dict(base_init)
    opt_v2 = optim.SGD(model_v2.parameters(), lr=lr)
    decider_v2 = DecisionMLP(2).to(device)
    opt_decider_v2 = optim.Adam(decider_v2.parameters(), lr=scheduler_lr)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir='experiments/image_classification/result/runs/iterbatch_exp')  # 日志输出在 image_classification/result/runs/iterbatch_exp

    # ---- Training Loop ----
    criterion = nn.CrossEntropyLoss()
    accs_base, losses_base = [], []
    accs_v1, losses_v1 = [], []
    accs_v2, losses_v2 = [], []
    for epoch in range(1, epochs+1):
        model_base.train()
        model_v1.train()
        model_v2.train()
        decider_v1.train()
        decider_v2.train()
        recent_losses_v2 = deque(maxlen=window_size)
        # Metrics
        loss_sum_base, correct_base, total_base = 0, 0, 0
        loss_sum_v1, correct_v1, total_v1 = 0, 0, 0
        loss_sum_v2, correct_v2, total_v2 = 0, 0, 0
        # For policy gradient (optional, can be extended)
        log_probs_v1, rewards_v1 = [], []
        log_probs_v2, rewards_v2 = [], []
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
            # --- Loss-only Decision ---
            opt_v1.zero_grad()
            out_v1 = model_v1(data)
            loss_v1 = criterion(out_v1, target)
            loss_v1.backward()
            opt_v1.step()
            loss_sum_v1 += loss_v1.item()
            pred_v1 = out_v1.argmax(dim=1)
            correct_v1 += (pred_v1 == target).sum().item()
            total_v1 += target.size(0)
            # Decision
            decider_input_v1 = torch.tensor([[loss_v1.item()]], dtype=torch.float32, device=device)
            action_probs_v1 = decider_v1(decider_input_v1)
            m_v1 = torch.distributions.Categorical(action_probs_v1)
            action_v1 = m_v1.sample()
            log_probs_v1.append(m_v1.log_prob(action_v1))
            if action_v1.item() == 1:
                # repeat
                opt_v1.zero_grad()
                out_v1 = model_v1(data)
                loss_v1 = criterion(out_v1, target)
                loss_v1.backward()
                opt_v1.step()
            # --- Context Decision ---
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
            decider_input_v2 = torch.tensor([[loss_v2.item(), avg_loss_v2]], dtype=torch.float32, device=device)
            action_probs_v2 = decider_v2(decider_input_v2)
            m_v2 = torch.distributions.Categorical(action_probs_v2)
            action_v2 = m_v2.sample()
            log_probs_v2.append(m_v2.log_prob(action_v2))
            if action_v2.item() == 1:
                opt_v2.zero_grad()
                out_v2 = model_v2(data)
                loss_v2 = criterion(out_v2, target)
                loss_v2.backward()
                opt_v2.step()
            recent_losses_v2.append(loss_v2.item())
        # ---- Epoch End: Evaluate ----
        acc_base, test_loss_base = evaluate(model_base, test_loader, device)
        acc_v1, test_loss_v1 = evaluate(model_v1, test_loader, device)
        acc_v2, test_loss_v2 = evaluate(model_v2, test_loader, device)
        print(f"Epoch {epoch}: Baseline acc={acc_base:.4f} loss={test_loss_base:.4f} | LossNet acc={acc_v1:.4f} loss={test_loss_v1:.4f} | CtxNet acc={acc_v2:.4f} loss={test_loss_v2:.4f}")
        # ---- TensorBoard Logging ----
        writer.add_scalar('Baseline/Accuracy', acc_base, epoch)
        writer.add_scalar('Baseline/Loss', test_loss_base, epoch)
        writer.add_scalar('LossNet/Accuracy', acc_v1, epoch)
        writer.add_scalar('LossNet/Loss', test_loss_v1, epoch)
        writer.add_scalar('CtxNet/Accuracy', acc_v2, epoch)
        writer.add_scalar('CtxNet/Loss', test_loss_v2, epoch)
        # ---- For Matplotlib ----
        accs_base.append(acc_base)
        losses_base.append(test_loss_base)
        accs_v1.append(acc_v1)
        losses_v1.append(test_loss_v1)
        accs_v2.append(acc_v2)
        losses_v2.append(test_loss_v2)
    writer.close()
    print('Training complete. Use TensorBoard to view results.')

    # ---- Matplotlib 可视化 ----
    # 图表保存路径改为 image_classification/result 目录下
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accs_base, 'o-', label='Baseline')
    plt.plot(epochs_range, accs_v1, 's-', label='LossNet')
    plt.plot(epochs_range, accs_v2, '^-', label='CtxNet')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, losses_base, 'o-', label='Baseline')
    plt.plot(epochs_range, losses_v1, 's-', label='LossNet')
    plt.plot(epochs_range, losses_v2, '^-', label='CtxNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss vs. Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('experiments/image_classification/result/iterbatch_exp_results.png')
    plt.show()

if __name__ == '__main__':
    main()