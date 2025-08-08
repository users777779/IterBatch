import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import LlamaTokenizer


class TextGenerationDataset(Dataset):
    """文本生成数据集类"""
    def __init__(self, texts, tokenizer, max_length=512):
        """初始化数据集

        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 序列最大长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        text = self.texts[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 返回input_ids和attention_mask
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class TextDataLoader:
    """文本数据加载器类"""
    def __init__(self, dataset_name='wikitext', tokenizer_name='huggyllama/llama-7b', batch_size=4, max_length=512):
        """初始化数据加载器

        Args:
            dataset_name: 数据集名称 ('wikitext' 或 'dialogue')
            tokenizer_name: 分词器名称
            batch_size: 批次大小
            max_length: 序列最大长度
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 加载分词器
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        # 添加pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载数据集
        if dataset_name == 'wikitext':
            # 加载WikiText-103数据集
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
            # 合并所有文本
            train_texts = [text for text in dataset['train']['text'] if len(text.strip()) > 0]
            val_texts = [text for text in dataset['validation']['text'] if len(text.strip()) > 0]
            test_texts = [text for text in dataset['test']['text'] if len(text.strip()) > 0]
        elif dataset_name == 'dialogue':
            # 这里可以加载对话数据集，例如PersonaChat或DialyDialog
            # 由于这些数据集可能需要额外的处理，这里提供一个占位符实现
            raise NotImplementedError("对话数据集加载功能需要根据具体数据集实现")
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 创建数据集对象
        self.train_dataset = TextGenerationDataset(train_texts, self.tokenizer, max_length)
        self.val_dataset = TextGenerationDataset(val_texts, self.tokenizer, max_length)
        self.test_dataset = TextGenerationDataset(test_texts, self.tokenizer, max_length)
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_loader(self):
        """获取验证数据加载器"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)