#  训练超参的解释可参考 https://zhuanlan.zhihu.com/p/363670628
#                   https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/4_trainer.html

#多卡训练
#accelerate launch \
#    --num_processes 2 \
#    --config_file sft.yaml\
#    --deepspeed_multinode_launcher standard fine-tune.py  \   #运行的文件
#    --data_path "./data/kst_train_cot.json" \                 #微调使用的数据
#    --output_dir "./lora_model" \                             #输出模型的路径
#    --model_name_or_path "../baichuan2/Baichuan2-7B-Base" \   #要微调的模型路径
#    --model_max_length 512 \
#    --num_train_epochs 3 \
#    --per_device_train_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --save_strategy epoch \
#    --learning_rate 2e-5 \
#    --lr_scheduler_type constant \
#    --adam_beta1 0.9 \
#    --adam_beta2 0.98 \
#    --adam_epsilon 1e-8 \
#    --max_grad_norm 1.0 \
#    --weight_decay 1e-4 \
#    --warmup_ratio 0.0 \
#    --logging_steps 1 \
#    --gradient_checkpointing True \
#    --bf16 True \
#    --tf32 True \
#    --use_lora True \
##########################################################
#单卡训练
deepspeed fine-tune.py \
    --data_path "./data/kst_train_cot.json" \
    --output_dir "./lora_model" \
    --model_name_or_path "../baichuan2/Baichuan2-7B-Base" \
    --model_max_length 512 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --bf16 True \
    --tf32 True \
    --use_lora True \
