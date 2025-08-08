# 文本生成实验结果汇总（最新）

配置：LLaMA-7B + LoRA（8-bit + CPU offload），WikiText-103，1 epoch，bs=1，max_train/eval_batches=50，loss_threshold=1.0，log_every_n_steps=1。

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

TensorBoard：experiments/text_generation/results/tensorboard/wikitext_<timestamp>/
指标npy：experiments/text_generation/results/*_metrics.npy
