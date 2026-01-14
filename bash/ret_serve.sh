python src/retriever/ret_serve.py \
    --faiss_index_path embedding/wiki.index \
    --corpus_jsonl_path embedding/wiki.jsonl \
    --emb_url http://localhost:65501/v1 \
    --emb_model qwen3-emb \
    --gpu_ids 4,5 \
    --use_gpu True \
    --host 127.0.0.1 \
    --port 8000


