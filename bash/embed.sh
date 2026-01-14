python src/retriever/embed.py \
    --model_path Qwen3-Embedding-0.6B \
    --corpus_path evaluation_dataset/wiki.jsonl \
    --embedding_path embedding/wiki.npy \
    --batch_size 4 \
    --gpu_ids 0,1,2,3