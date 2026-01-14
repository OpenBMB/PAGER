import os
import faiss
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Direct batch vLLM page generator")
parser.add_argument(
    "--embedding_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--index_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--index_chunk_size",
    type=int,
    default=50000,
)
parser.add_argument(
    "--faiss_use_gpu",
    type=bool,
    default=True,
)
args = parser.parse_args()
print(args)

# config
# embedding_path = "/home/yyk/yyk08/CacheNote_HX/1016_cache_limit/embed/sqa_corpus_qwen3_8b_sample_10000.npy"
# index_path = "/home/yyk/yyk08/CacheNote_HX/1016_cache_limit/index/sqa_corpus_qwen3_8b_sample_10000.index"
# index_chunk_size = 50000
# faiss_use_gpu = True
embedding_path=args.embedding_path
index_path = args.index_path
index_chunk_size = args.index_chunk_size
faiss_use_gpu = args.faiss_use_gpu

# do not change below
os.makedirs(os.path.dirname(index_path), exist_ok=True)

embedding = np.load(embedding_path)
dim = embedding.shape[1]
vec_ids = np.arange(embedding.shape[0]).astype(np.int64)

cpu_flat = faiss.IndexFlatIP(dim)
cpu_index = faiss.IndexIDMap2(cpu_flat)

total = embedding.shape[0]
print(f"Start building FAISS index, total vectors: {total}")

with tqdm(
    total=total,
    desc="[faiss] Indexing: ",
    unit="vec",
) as pbar:
    for start in range(0, total, index_chunk_size):
        end = min(start + index_chunk_size, total)
        cpu_index.add_with_ids(embedding[start:end], vec_ids[start:end])
        pbar.update(end - start)

faiss.write_index(cpu_index, index_path)
print("[faiss] Indexing success.")
print(f"Saved index to {index_path}")
