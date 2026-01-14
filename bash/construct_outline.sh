python src/construct_outline.py \
    --model_name /home/yyk/yyk08/qwen32b \
    --gpu_ids 2,3 \
    --input_file nq_test_shuffle_2k.jsonl \
    --out_file outline/outline_nq_test_shuffle_2k.jsonl \
    --max_iters 10 \
    --batch_size 2000 \
    --sample_limit 10 \
    --seed 66