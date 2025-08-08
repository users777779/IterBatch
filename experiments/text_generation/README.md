# 文本生成实验（LLaMA + LoRA）

## 实验目标
- 任务：自回归文本生成（Causal LM）
- 模型：LLaMA + LoRA（仅训练 LoRA 权重）
- 数据：WikiText-103 或 DailyDialog 对话
- 指标：困惑度 Perplexity（越低越好）
- 脏数据处理：基于历史损失统计跳过异常批次（outliers）

## 实现概述
- `model.py`
  - 使用 HuggingFace Transformers + PEFT（LoRA）。
  - 8-bit 量化 + CPU offload：`BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)`；`device_map='auto'`。
  - Tokenizer 兼容：无 `pad_token` 时以 `eos_token` 代替；`padding_side='left'`。
  - `resize_token_embeddings` 同步词表；训练时 `config.use_cache=False`。
  - 仅对 LoRA 权重 `requires_grad=True`，其余权重冻结，降低显存与反传开销。
- `data_loader.py`
  - 加载并拼接文本，按 `max_length` 分块，构造 `input_ids/attention_mask/labels`。
- `strategy.py`
  - 基类：维护 `loss_history`，outlier 判定为 `> mean + 2.5*std`；`evaluate()` 计算 PPL。
  - 设备适配：若使用 `device_map`/offload，不手动将 batch `.to(device)`，由 accelerate 接管，避免 meta/cuda 冲突。
  - Baseline：每批训练 1 次；
  - ABR：基于初始损失的重复训练；默认阈值 `loss_threshold=1.0`；`repeats = min(int(init_loss/threshold), max_repeats)`，不足阈值则为 1；
  - Learnable：调度器（两层 MLP）根据 `[loss, ppl]` 预测 1..5 次重复，并在线学习；
  - SlidingWindow：基于最近窗口的损失斜率/波动决定重复次数。
- `train.py`
  - 目录归一化：`--save_dir` 相对路径按脚本目录解析，确保结果写入 `experiments/text_generation/results/`。
  - 预热“药引子”：`--primer_batches` 可先用少量高质量批次稳定损失统计。
  - 快速试跑：`--max_train_batches`、`--max_eval_batches` 限制批次数。
  - TensorBoard：新增“步级”日志 `--log_every_n_steps`（默认每步记录），支持查看 Step 级 Loss/PPL/Repeats/LR 曲线。

## 目录结构
```
text_generation/
├── README.md
├── environment.yml
├── model.py
├── data_loader.py
├── strategy.py
├── train.py
└── results/
    ├── baseline_metrics.npy
    ├── abr_metrics.npy
    ├── learnable_metrics.npy
    ├── window_metrics.npy
    └── tensorboard/
        └── wikitext_<timestamp>/
```

## 运行方式（Linux 示例）
- 单策略：
```
python experiments/text_generation/train.py \
  --model llama --model_name huggyllama/llama-7b \
  --dataset wikitext --strategies abr \
  --epochs 1 --batch_size 1 --max_length 512 \
  --max_train_batches 50 --max_eval_batches 50 \
  --loss_threshold 1.0 --max_repeats 5 \
  --log_every_n_steps 1 --save_dir results
```
- 四策略依次运行（避免同进程显存碎片导致 OOM）：
```
for s in baseline abr learnable window; do \
  python experiments/text_generation/train.py --model llama --model_name huggyllama/llama-7b \
  --dataset wikitext --strategies $s --epochs 1 --batch_size 1 --max_length 512 \
  --max_train_batches 50 --max_eval_batches 50 --log_every_n_steps 1 \
  --loss_threshold 1.0 --save_dir results || break; done
```
- 查看 TensorBoard：
```
tensorboard --logdir experiments/text_generation/results/tensorboard --port 6006 --bind_all
```
将 X-Axis 设为 Step，可见步级曲线；在 Filter 中输入 `Step/` 查看步级指标。

## 最新一次四策略对比（1 epoch, bs=1, max_train/eval_batches=50, 阈值=1.0）
- baseline
  - Train loss 2.2239, PPL 10.7744
  - Val loss 1.7085, PPL 5.5207
  - Test loss 1.8713, PPL 6.4968
- abr（loss_threshold=1.0）
  - Train loss 1.9838, PPL 8.0007
  - Val loss 1.6574, PPL 5.2454
  - Test loss 1.8007, PPL 6.0541
  - Avg repeats 1.46
- learnable
  - Train loss 2.2205, PPL 10.8530
  - Val loss 1.7223, PPL 5.5972
  - Test loss 1.8826, PPL 6.5703
  - Avg repeats 1.00
- window
  - Train loss 1.8883, PPL 7.2054
  - Val loss 1.8970, PPL 6.6660
  - Test loss 1.8635, PPL 6.4465
  - Avg repeats 3.28

## 经验与注意事项
- 若使用 `device_map='auto'` 与 offload，不要对整体模型调用 `.to(device)`，也不要在前向前手动将 batch 强制搬到单一设备，避免 meta/cuda 冲突。
- ABR 的阈值需与损失尺度匹配。阈值过高会导致重复次数≈1；可调低阈值或改为上取整/相对阈值。
- 显存吃紧时：分策略分进程运行、减小 `max_length`/`batch_size`、启用 8-bit + CPU offload。
- 结果写入脚本目录下的 `results/`，便于跨目录调用的一致性。