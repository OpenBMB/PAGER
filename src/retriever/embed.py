from infinity_emb import EngineArgs, AsyncEngineArray
from tqdm import tqdm
import jsonlines
import numpy as np
import os
import gc
import asyncio
import argparse

parser = argparse.ArgumentParser(description="Direct batch vLLM page generator")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--corpus_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--embedding_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--gpu_ids",
    type=str,
    default="0,1,2,3",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)
args = parser.parse_args()
print(args)

# config
# model_path = "/home/yyk/yyk08/models/Qwen3-Embedding-0.6B"
# corpus_path = "/home/yyk/yyk08/CacheNote_HX/1016_cache_limit/sqa_corpus_qwen3_8b_sample_10000.jsonl"
# embedding_path = "/home/yyk/yyk08/CacheNote_HX/1016_cache_limit/embed/sqa_corpus_qwen3_8b_sample_10000.npy"
# batch_size = 4
# gpu_ids = "0,1,2,3"
model_path = args.model_path
corpus_path = args.corpus_path
embedding_path = args.embedding_path
batch_size = args.batch_size 
gpu_ids = args.gpu_ids

# do not change below
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

infinity_engine_args = EngineArgs(
    model_name_or_path=model_path,
    batch_size=batch_size,
    bettertransformer=False,
    pooling_method="auto",
    device="cuda",
    model_warmup=False,
    trust_remote_code=True,
)
model = AsyncEngineArray.from_args([infinity_engine_args])[0]

contents = []
with jsonlines.open(corpus_path, mode="r") as reader:
    for i, item in enumerate(reader):
        contents.append(item["contents"])

data = contents


async def embed():
    async with model:
        eff_bs = batch_size * len(gpu_ids.split(","))
        n = len(data)
        pbar = tqdm(total=n, desc="[infinity] Embedding: ")
        embeddings = []
        for i in range(0, n, eff_bs):
            chunk = data[i : i + eff_bs]
            vecs, _ = await model.embed(sentences=chunk)
            embeddings.extend(vecs)
            pbar.update(len(chunk))
        pbar.close()

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(embedding_path, embeddings)
    print(f"Saved embedding to {embedding_path}")
    del embeddings
    gc.collect()


async def main():
    await embed()


if __name__ == "__main__":
    asyncio.run(main())
