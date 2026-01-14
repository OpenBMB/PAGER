python src/infer_page.py \
    --model /home/yyk/yyk08/qwen32b \
    --input_file output_data/page/nq_test_shuffle_2k.jsonl \
    --output_file output_data/infer/nq_test_shuffle_2k.jsonl \
    --batch_size 2000 \
    --gpu_ids 0,1