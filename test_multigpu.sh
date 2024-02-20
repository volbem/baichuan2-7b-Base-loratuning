#accelerate launch \
#    --num_processes 2\
#    --config_file sft.yaml\
#    --deepspeed_multinode_launcher standard test.py \
#    --model_path ../baichuan2/Baichuan2-7B-Base \
#    --data_path ./data/kst_test_cot.json \
#    --out_file ./results_cot_split/output_multigpu.json \
#    --batch_size 4
#
# deepspeed使用ZeRO， 是一种针对大规模分布式深度学习的新型内存优化技术
# 在sft.yaml文件的deepspeed_config中的参数解释可参考   https://zhuanlan.zhihu.com/p/639927499
# 其他超参可参考  https://blog.csdn.net/qq_56591814/article/details/134390073
#              https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/7_accelerate.html
accelerate launch \
    --num_processes 2\
    --config_file sft.yaml\
    --deepspeed_multinode_launcher standard test.py \
    --model_path ../baichuan2/Baichuan2-7B-Base \
    --data_path ./data/kst_test_cot.json \
    --out_file ./results_cot_split/output_multigpu.json \
    --batch_size 4
