python src/construct_page.py \
    --model_name /home/yyk/yyk08/qwen32b \
    --gpu_ids 2,3 \
    --retrieval_url http://g71:5005 \
    --input_file output_data/outline/new_outline_nq.jsonl \
    --out_file output_data/page/nq_test_shuffle_2k.jsonl \
    --max_iters 10 \
    --batch_size 2000 \
    --sample_limit 10 \
    --seed 66

