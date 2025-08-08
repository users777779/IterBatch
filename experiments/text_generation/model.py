import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


class LLaMAWithLoRA(nn.Module):
    """LLaMA + LoRA 封装
    - 负责加载分词器与模型
    - 应用LoRA到LLaMA的注意力与MLP投影层
    - forward 兼容 HF CausalLM (input_ids, attention_mask, labels)
    """
    def __init__(self, model_name: str, lora_rank: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # LLaMA通常无pad_token，设置为eos保持因果LM一致
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # 生成式任务常用left padding

        # 加载基础因果语言模型（优先8-bit量化，其次FP16 + 自动设备映射）
        use_cuda = torch.cuda.is_available()
        device_map = 'auto' if use_cuda else None
        if use_cuda:
            # 使用8-bit量化 + FP32 CPU offload，兼顾显存与可加载性
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=quant_config,
            )
        else:
            dtype = torch.float32
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=dtype,
            )

        self.backbone.resize_token_embeddings(len(self.tokenizer))
        # 训练时禁用use_cache
        if hasattr(self.backbone, 'config'):
            self.backbone.config.use_cache = False

        # LoRA 配置：覆盖注意力和MLP常见投影层
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # 显式冻结非LoRA参数，仅训练LoRA权重
        for name, param in self.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)


class SchedulerLLaMA(nn.Module):
    """简单的调度器：输入(batch初始loss, perplexity)，输出建议的重复训练次数(实数)"""
    def __init__(self, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        # 输入维度=2: [loss, ppl]
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
