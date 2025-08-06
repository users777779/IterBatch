import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    """简单的CNN模型，用于图像分类"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def clone(self):
        """创建模型的深拷贝"""
        return SimpleCNN(self.fc2.out_features)


class ResNet18(nn.Module):
    """ResNet-18模型，用于图像分类"""
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # 修改最后一层全连接层以适应不同的类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def clone(self):
        """创建模型的深拷贝"""
        clone_model = ResNet18(self.model.fc.out_features)
        clone_model.load_state_dict(self.state_dict())
        return clone_model


class Transformer(nn.Module):
    """简单的Transformer模型，用于图像分类"""
    def __init__(self, num_classes=10, img_size=32, patch_size=4, d_model=64, nhead=4, num_layers=3):
        super(Transformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        # 图像分块和线性投影
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        # 类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # 分块和投影
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, num_patches, d_model

        # 添加类别标记
        cls_token = self.cls_token.expand(B, -1, -1)  # B, 1, d_model
        x = torch.cat([cls_token, x], dim=1)  # B, num_patches + 1, d_model

        # 添加位置编码
        x = x + self.pos_embed

        # Transformer编码
        x = self.transformer_encoder(x)

        # 取类别标记的输出进行分类
        x = x[:, 0]
        x = self.fc(x)

        return x

    def clone(self):
        """创建模型的深拷贝"""
        clone_model = Transformer(
            num_classes=self.fc.out_features,
            img_size=self.img_size,
            patch_size=self.patch_size,
            d_model=self.d_model,
            nhead=self.transformer_encoder.layers[0].self_attn.num_heads,
            num_layers=len(self.transformer_encoder.layers)
        )
        clone_model.load_state_dict(self.state_dict())
        return clone_model


def get_model(model_name, num_classes=10):
    """获取指定的模型"""
    if model_name.lower() == 'cnn':
        return SimpleCNN(num_classes)
    elif model_name.lower() == 'resnet18':
        return ResNet18(num_classes)
    elif model_name.lower() == 'transformer':
        return Transformer(num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")