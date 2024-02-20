
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_path ../baichuan2/Baichuan2-7B-Base \
    --data_path ./data/kst_test_cot.json \
    --out_file ./results_cot_split/output_singlegpu.json \
    --batch_size 4 \
    --max_new_tokens 128 \

    