nohup accelerate launch --config_file sft.yaml \
    --deepspeed_multinode_launcher standard test.py \
    --model_path /mntcephfs/data/med/jiangfeng/KST/Baichuan2/cot_split \
    --data_path ./data/kst_test_cot.json \
    --out_file ./results_cot_split/output.json \
    > inference-cot_split.log &
