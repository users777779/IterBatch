# IterBatch: 自适应批次重复训练策略

## 项目概述
自适应批次重复训练 (ABR) 策略是一种根据实时损失动态调整批次训练频率的优化方法。该策略使模型能够自适应地聚焦于高损失的困难样本，从而提升训练效率和模型性能。

## 实验设计
本实验通过对比三种训练策略来验证ABR策略的有效性：
1. **基线策略**：每个批次只训练一次
2. **ABR策略**：根据损失动态调整批次的重复训练次数
3. **固定重复策略**：每个批次固定重复训练2次

实验使用加州房价预测数据集和MLP模型，主要评估指标包括均方误差(MSE)、均方根误差(RMSE)和平均绝对误差(MAE)。

## 项目结构
```
IterBatch/
├── README.md                 # 项目主文档
├── docs/
│   └── Experimental_Log_and_Paper_Guide.markdown
└── experiments/              # 实验目录
    ├── image_classification/  # 图像分类实验 (@zjh)
    │   ├── README.md
    │   ├── environment.yml
    │   └── model.py
    ├── regression/            # 回归任务实验 (@gsz)
    │   ├── README.md
    │   ├── environment.yml
    │   ├── data_loader.py
    │   ├── model.py
    │   ├── strategy.py
    │   └── train.py
    ├── image_captioning/      # 图像描述实验 (@zjh)
    │   ├── README.md
    │   └── environment.yml
    └── text_generation/       # 文本生成实验 (@gsz)
        ├── README.md
        └── environment.yml
```

## 实验分工
- **图像分类 & 图像描述**：@zjh
- **回归任务 & 文本生成**：@gsz

## 环境要求与依赖管理
- 推荐使用Anaconda或Miniconda创建虚拟环境
- 每个实验目录下包含独立的依赖要求和安装说明
- 具体依赖信息请查看各实验目录下的README.md

## 运行实验
每个实验都是独立的，具体实现和执行方式请查看对应实验目录下的README.md：
- [图像分类实验](experiments/image_classification/README.md)
- [回归任务实验](experiments/regression/README.md)
- [图像描述实验](experiments/image_captioning/README.md)
- [文本生成实验](experiments/text_generation/README.md)

## 实验结果
实验结果将保存在 `results/` 目录下，包括：
- 训练和测试指标的历史记录
- 损失曲线和RMSE曲线的可视化
- ABR策略的批次训练历史分析

## 关于ABR策略
自适应批次重复训练 (ABR) 策略是一种根据实时损失动态调整批次训练频率的优化方法。该策略使模型能够自适应地聚焦于高损失的困难样本，从而提升训练效率和模型性能。

### 核心机制
- 根据每个批次的初始损失值动态决定重复训练次数
- 损失越高，重复训练次数越多（最多不超过设定的最大值）
- 记录每个批次的训练历史，包括损失变化和重复次数

## 预期结论
通过实验，我们期望验证ABR策略在以下方面的优势：
1. 提高模型收敛速度
2. 提升最终模型性能
3. 增强模型对困难样本的学习能力
4. 相比固定重复策略更高效地利用计算资源
