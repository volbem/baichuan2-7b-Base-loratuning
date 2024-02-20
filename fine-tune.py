import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, model_max_length, user_tokens=[195], assistant_tokens=[196]):
        """
        初始化 SupervisedDataset 类的实例。

        参数:
            data_path: 数据文件的路径。
            tokenizer: 用于文本编码的分词器。
            model_max_length: 模型能处理的最大文本长度。
            user_tokens: 表示用户输入的特殊标记的 ID 列表，默认为 [195]。
            assistant_tokens: 表示助手回复的特殊标记的 ID 列表，默认为 [196]。
        """
        super(SupervisedDataset, self).__init__()  # 调用父类的构造器
        self.data = json.load(open(data_path))  # 从指定路径加载数据
        self.tokenizer = tokenizer  # 设置分词器
        self.model_max_length = model_max_length  # 设置模型的最大文本长度
        self.user_tokens = user_tokens  # 设置用户输入的特殊标记
        self.assistant_tokens = assistant_tokens  # 设置助手回复的特殊标记
        self.ignore_index = -100  # 设置忽略索引值，通常用于标记填充或无效数据

        # 对数据集中的第一个元素进行预处理并打印结果
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))  # 解码并打印输入 ID

        # 提取有效标签并打印
        labels = [id_ for id_ in item["labels"] if id_ != self.ignore_index]
        print("label:", self.tokenizer.decode(labels))  # 解码并打印标签

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        # 初始化输入 ID 和标签列表
        input_ids = []
        labels = []

        # 遍历对话中的每条消息
        for message in example["conversations"]:
            from_ = message["from"]  # 消息来源（human 或 assistant）
            value = message["value"]  # 消息内容
            value_ids = self.tokenizer.encode(value)  # 将消息内容编码为 ID

            # 根据消息来源处理输入 ID 和标签
            if from_ == "human":
                # 如果消息来自人类用户
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:
                # 如果消息来自助手
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids

        # 在序列末尾添加结束符
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)

        # 裁剪或填充输入 ID 和标签到模型最大长度
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))

        # 将列表转换为 PyTorch 张量
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)

        # 创建注意力掩码，标记哪些位置是填充的
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 返回包含输入 ID、标签和注意力掩码的字典
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
