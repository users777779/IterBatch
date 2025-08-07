import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

class HousingDataLoader:
    """房价数据集加载器"""
    def __init__(self, dataset_name='california', batch_size=5, test_size=0.2):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.load_data()

    def load_data(self):
        """加载并预处理数据集"""
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

        # 划分训练集和测试集
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int((1 - self.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

    def get_train_loader(self):
        """获取训练数据加载器"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)