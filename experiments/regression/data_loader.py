"""数据加载与预处理

本模块提供 `HousingDataLoader`，用于：
- 下载/加载 California Housing 数据集
- 标准化特征（`StandardScaler`）
- 根据 `test_size` 与 `val_ratio` 划分 train/val/test
- 以 `torch.utils.data.DataLoader` 的形式对外提供迭代器

关键约定：
- 仅实现 `california` 数据集；传入其它名称会抛出错误
- 当 `val_ratio == 0.0` 时，不生成验证集，`get_val_loader()` 返回 `None`
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

class HousingDataLoader:
    """房价数据集加载器

    支持 train/val/test 划分，其中验证集通过 `val_ratio` 控制，可选。
    """
    def __init__(self, dataset_name='california', batch_size=5, test_size=0.2, val_ratio=0.0):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_ratio = val_ratio
        self.scaler = StandardScaler()
        self.load_data()

    def load_data(self):
        """加载并预处理数据集

        流程：
        1) 从 sklearn 获取特征与标签
        2) 对特征做标准化（均值0方差1）
        3) 转换为 `torch.Tensor`
        4) 按比例拆分为 train/val/test
        """
        if self.dataset_name.lower() == 'california':
            dataset = fetch_california_housing()
        else:
            raise ValueError("不支持的数据集: {}".format(self.dataset_name))

        X, y = dataset.data, dataset.target

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 转换为张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # 划分训练集/验证集/测试集
        dataset = TensorDataset(X_tensor, y_tensor)
        total_len = len(dataset)
        test_count = int(self.test_size * total_len)
        # 验证集比例基于整体样本数，避免过拟合测试集
        val_count = int(self.val_ratio * total_len)
        train_count = total_len - test_count - val_count
        if train_count <= 0:
            raise ValueError("train/val/test 划分无效：训练集大小需大于 0，请调整 test_size/val_ratio。")

        splits = [train_count, val_count, test_count] if val_count > 0 else [train_count, test_count]
        subsets = random_split(dataset, splits)
        if val_count > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = subsets
        else:
            self.train_dataset, self.test_dataset = subsets
            self.val_dataset = None

    def get_train_loader(self):
        """获取训练数据加载器（包含打乱）"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        """获取验证数据加载器（可能为 None）"""
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)