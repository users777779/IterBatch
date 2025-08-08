import math
from typing import Dict, Literal

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class TextDataLoader:
    """
    - 支持 WikiText-103 与 DailyDialog 两类数据
    - 使用 tokenizer 对文本进行拼接+固定block_size切块，返回因果LM训练样本
    - 提供 train/val/test DataLoader
    """
    def __init__(self, dataset_name: Literal['wikitext', 'dialogue'], tokenizer_name: str,
                 batch_size: int = 4, max_length: int = 512):
        # 直接加载分词器，避免加载完整模型以节省显存/内存
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.batch_size = batch_size
        self.block_size = max_length

        if dataset_name == 'wikitext':
            raw = load_dataset('wikitext', 'wikitext-103-v1')
            # 字段为 'text'
            split_map = {
                'train': raw['train'],
                'validation': raw['validation'],
                'test': raw['test']
            }
        elif dataset_name == 'dialogue':
            # 使用 DailyDialog 作为示例对话数据集
            raw = load_dataset('daily_dialog')
            # 将 utterances 拼接为单条文本
            def join_dialogue(example):
                # 每个对话：list[str]
                text = '\n'.join(example['dialog']) if 'dialog' in example else '\n'.join(example['utterances'])
                return {'text': text}
            split_map = {
                'train': raw['train'].map(join_dialogue, remove_columns=raw['train'].column_names),
                'validation': raw['validation'].map(join_dialogue, remove_columns=raw['validation'].column_names),
                'test': raw['test'].map(join_dialogue, remove_columns=raw['test'].column_names),
            }
        else:
            raise ValueError(f'未知数据集: {dataset_name}')

        # 统一tokenize与分块
        def tokenize_fn(examples: Dict[str, list]):
            return self.tokenizer(examples['text'], truncation=False)

        tokenized = {}
        for k, ds in split_map.items():
            tokenized[k] = ds.map(
                tokenize_fn,
                batched=True,
                remove_columns=ds.column_names,
                desc=f'Tokenizing {k}'
            )

        def group_texts(examples: Dict[str, list]):
            # 拼接为长序列再切分固定块
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated['input_ids'])
            total_length = (total_length // self.block_size) * self.block_size
            result = {}
            for k, t in concatenated.items():
                t = t[:total_length]
                # 切块
                result[k] = [t[i:i + self.block_size] for i in range(0, total_length, self.block_size)]
            result['labels'] = result['input_ids'].copy()
            return result

        grouped = {}
        for k, ds in tokenized.items():
            grouped[k] = ds.map(
                group_texts,
                batched=True,
                desc=f'Grouping {k}'
            )

        # 设置 PyTorch 张量格式由 DataLoader 处理，保持字典键
        self.train_set = grouped['train']
        self.val_set = grouped['validation']
        self.test_set = grouped['test']

        def collate_fn(features):
            input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
            labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]
            input_ids = torch.stack(input_ids, dim=0)
            labels = torch.stack(labels, dim=0)
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        # DataLoader
        self._train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self._val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self._test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def get_train_loader(self):
        return self._train_loader

    def get_val_loader(self):
        return self._val_loader

    def get_test_loader(self):
        return self._test_loader
