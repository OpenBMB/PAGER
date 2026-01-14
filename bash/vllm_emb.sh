CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen3-emb \
    --model Qwen3-Embedding-0.6B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 65501 \
    --task embed \
    --gpu-memory-utilization 0.6