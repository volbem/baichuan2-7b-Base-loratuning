# nohup python scorer.py --test_file results_original/output.json --output_dir results_original/ > eval_original.log &
# nohup python scorer.py --test_file results_cot/output_clear.json --output_dir results_cot/ > eval_cot.log &
# nohup python scorer.py --test_file results_action/output_clear.json --output_dir results_action/ > eval_action.log &
# nohup python scorer.py --test_file results_cot_split/output_clear.json --output_dir results_cot_split/ --is_action True > eval_cot_split.log &
python scorer.py \
    --test_file ../results_cot_split/output_clear.json \
    --output_dir ../results_cot_cosine \
    --is_action True