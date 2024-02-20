# baichuan2-7b-Base-loratuning


### 安装

1. huggingface高速下载 https://mp.weixin.qq.com/s/Fx6nfFt_RPwDHZ3V73PD1Q
```bash
git clone https://github.com/LetheSec/HuggingFace-Download-Accelerator.git
cd HuggingFace-Download-Accelerator
```

2. 下载baichuan2 7b base模型
```bash
python hf_download.py --model baichuan-inc/Baichuan2-7B-Base --save_dir ./Baichuan2-7B-Base
```

3. 下载微调指令数据(以BelleGroup/train_0.5M_CN指令集为示例)
```bash
python hf_download.py --dataset BelleGroup/train_0.5M_CN --save_dir ./data
```

4. 安装依赖
```bash
pip install -r requirements.txt
```

### 微调
运行train.sh进行微调
```bash
sh train.sh
#或使用nohup在后台运行 运行过程记录到train.log
#nohup sh -u train.sh > train.log 2>&1 &
```

### 预测
运行test.sh进行预测
```bash
sh test.sh
#或使用nohup在后台运行 运行过程记录到test.log
#nohup sh -u test.sh > test.log 2>&1 &
```

### 对微调预测结果进行清理和评分(路径为./result)

1. 清理

```bash
python post_process.py
```

2. 评分

  使用HuggingFace-Download-Accelerator下载bert评分的模型(以microsoft/deberta-xlarge-mnli模型为示例)
```bash
python hf_download.py --model microsoft/deberta-xlarge-mnli --save_dir ./deberta-xlarge-mnli
```

  评分
```bash
sh scorer.sh
#或使用nohup在后台运行 运行过程记录到scorer.log
#nohup sh -u scorer.sh > scorer.log 2>&1 &
```

