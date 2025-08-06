# IterBatch 实验说明

本目录下的 `main.py` 实现了三种批训练策略的对比实验：

## 实验方案
1. **Baseline**：每个 batch 只训练一次，记录每轮平均 loss 和 accuracy。
2. **Loss-only 决策网络**：每个 batch 用当前 loss 作为输入，MLP 判决是否重复训练该 batch（重复一次）。
3. **Context 决策网络**：每个 batch 用当前 loss 和前 10 步 loss 均值作为输入，MLP 判决是否重复训练该 batch。

三种实验均使用相同的模型初始参数，保证公平性。每种实验有独立优化器。

## 主要参数
- 训练轮数（epoch）：7
- 批大小（batch_size）：64
- 主模型学习率：0.01
- 决策网络学习率：1e-4
- 滑动窗口大小（context）：10

## 可视化输出
- **TensorBoard**：所有实验的 loss/accuracy 曲线均写入 `result/runs/iterbatch_exp` 日志目录。
- **PNG 图表**：训练结束后自动生成 `result/iterbatch_exp_results.png`，对比三组实验的 loss 和 accuracy 曲线。

## 运行方法
1. 安装依赖（建议 Python 3.8+，需提前安装好 torch、torchvision、numpy、matplotlib、tensorboard）：
   ```bash
   pip install torch torchvision numpy==1.26.4 matplotlib tensorboard
   ```
2. 运行实验脚本：
   ```bash
   python main.py
   ```
3. 查看 TensorBoard 曲线：
   ```bash
   tensorboard --logdir=./result/runs
   ```
   浏览器访问 http://localhost:6006
4. 查看 PNG 图表：
   训练结束后会在 `result/` 目录生成 `iterbatch_exp_results.png`，可直接打开查看。

## 备注
- 若遇到 numpy 2.x 兼容性问题，请降级 numpy 至 1.26.4。
- MNIST 数据集会自动下载到 `data/` 目录。
- 如需自定义参数，可直接修改脚本内的超参数设置。
