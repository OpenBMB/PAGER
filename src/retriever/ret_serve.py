import os
import faiss
import asyncio
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import orjson
from tqdm import tqdm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
import argparse
import uvicorn
parser = argparse.ArgumentParser(description="Direct batch vLLM page generator")
parser.add_argument(
    "--faiss_index_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--corpus_jsonl_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--emb_url",
    type=str,
    default=None,
)
parser.add_argument(
    "--emb_model",
    type=str,
    default=None,
)
parser.add_argument(
    "--gpu_ids",
    type=str,
    default="4,5",
)
parser.add_argument(
    "--use_gpu",
    type=bool,
    default=True,
)
parser.add_argument(
    "--host", 
    type=str, 
    default=None, 
    help="Host for the API server"
)
parser.add_argument(
    "--port", 
    type=int, 
    default=None, 
    help="Port for the API server"
)
args = parser.parse_args()
print(args)


# -------------------- config --------------------
# FAISS_INDEX_PATH = "/home/yyk/yyk08/CacheNote_HX/0927_new_setting/PAGER/embedding/sqa_corpus_qwen3_8b_sample_10000.index"
# CORPUS_JSONL_PATH = "/home/yyk/yyk08/CacheNote_HX/1016_cache_limit/sqa_corpus_qwen3_8b_sample_10000.jsonl"
# EMB_URL = "http://localhost:65501/v1"
# EMB_MODEL = "qwen3-emb"
# MAX_TOPK = 999
# API_KEY = "None"
# GPU_IDS = "4,5"
# USE_GPU = True
FAISS_INDEX_PATH = args.faiss_index_path
CORPUS_JSONL_PATH = args.corpus_jsonl_path
EMB_URL = args.emb_url
EMB_MODEL = args.emb_model
MAX_TOPK = 999
API_KEY = "None"
GPU_IDS = args.gpu_ids
USE_GPU = args.use_gpu

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS


app = FastAPI(title="FAISS Retrieval Service", version="1.0.0")


# -------------------- schemas --------------------
class SearchRequest(BaseModel):
    queries: List[str] = Field(..., description="Query list")
    topk: int = Field(5, gt=0, description="Top-k results per query")

    @validator("queries")
    def queries_not_empty(cls, v):
        if not v:
            raise ValueError("queries cannot be empty")
        return v

    @validator("topk")
    def topk_limit(cls, v):
        if v > MAX_TOPK:
            raise ValueError(f"topk too large (>{MAX_TOPK})")
        return v


class SearchResponse(BaseModel):
    contents: List[List[Dict[str, Any]]]
    scores: List[List[float]]


# -------------------- globals --------------------
INDEX: faiss.Index = None
CORPUS: List[str] = []
CLIENT: AsyncOpenAI = None
INDEX_DIM: int = -1

EMB_SEMAPHORE = asyncio.Semaphore(8)
SEARCH_SEMAPHORE = asyncio.Semaphore(16)


def _load_corpus_jsonl(path: str) -> List[Dict[str, str]]:
    corpus: List[Dict[str, str]] = []
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        with tqdm(
            total=file_size,
            desc="Loading corpus",
            unit="B",
            unit_scale=True,
            ncols=100,
        ) as pbar:
            for raw_line in f:
                pbar.update(len(raw_line))
                line_data = orjson.loads(raw_line)

                _id = str(line_data.get("id", ""))
                _contents = line_data.get("contents", "")
                if not isinstance(_contents, str):
                    _contents = str(_contents)

                parts = _contents.split("\n", 1)
                title = parts[0].strip()
                text = parts[1] if len(parts) > 1 else ""

                corpus.append(
                    {
                        "id": _id,
                        "title": title,
                        "text": text,
                        "contents": _contents,
                    }
                )
    return corpus


def _load_faiss_index(path: str) -> faiss.Index:
    print(f"[startup] Loading FAISS index from {path} ...")
    index = faiss.read_index(path)
    print(f"[startup] Index loaded. ntotal={index.ntotal}, dim={index.d}")
    return index


async def _get_embeddings(queries: List[str]) -> np.ndarray:
    async with EMB_SEMAPHORE:
        resp = await CLIENT.embeddings.create(model=EMB_MODEL, input=queries)
        embs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        return np.vstack(embs)


async def _faiss_search(embs: np.ndarray, topk: int):
    async with SEARCH_SEMAPHORE:
        D, I = await asyncio.to_thread(INDEX.search, embs, topk)
        return D, I


def _build_results(
    I: np.ndarray,
    D: np.ndarray,
    corpus: Dict[str, Any],
) -> Dict[str, List[List[Any]]]:
    contents_batch, scores_batch = [], []
    N = len(corpus)
    for idxs, dists in zip(I, D):
        cur_c, cur_s = [], []
        for idx, dist in zip(idxs, dists):
            if idx == -1:
                continue
            if 0 <= idx < N:
                cur_c.append(corpus[idx])
                cur_s.append(float(dist))
        contents_batch.append(cur_c)
        scores_batch.append(cur_s)
    return {"contents": contents_batch, "scores": scores_batch}


# -------------------- startup --------------------
@app.on_event("startup")
def on_startup():
    global INDEX, CORPUS, CLIENT, INDEX_DIM

    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    INDEX = _load_faiss_index(FAISS_INDEX_PATH)
    INDEX_DIM = INDEX.d

    if not os.path.exists(CORPUS_JSONL_PATH):
        raise RuntimeError(f"Corpus jsonl not found: {CORPUS_JSONL_PATH}")
    CORPUS = _load_corpus_jsonl(CORPUS_JSONL_PATH)

    CLIENT = AsyncOpenAI(base_url=EMB_URL, api_key=API_KEY)

    if INDEX_DIM <= 0:
        raise RuntimeError("Invalid FAISS index dimension.")

    if USE_GPU:
        print(f"[startup] Moving index to GPU(s): {GPU_IDS}")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        INDEX = faiss.index_cpu_to_all_gpus(INDEX, co)

    print(
        f"[startup] index loaded: {FAISS_INDEX_PATH}, dim={INDEX_DIM}, use_gpu={USE_GPU}"
    )
    print(f"[startup] corpus loaded: {CORPUS_JSONL_PATH}, size={len(CORPUS)}")
    print(f"[startup] embedding url: {EMB_URL}, model: {EMB_MODEL}")


# -------------------- routes --------------------
@app.get("/health")
def health():
    ok = INDEX is not None and CORPUS and CLIENT is not None
    return {
        "status": "ok" if ok else "not_ready",
        "index_dim": INDEX_DIM,
        "corpus_size": len(CORPUS) if CORPUS else 0,
        "emb_url": EMB_URL,
        "emb_model": EMB_MODEL,
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        embs = await _get_embeddings(req.queries)
        if embs.shape[1] != INDEX_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dim {embs.shape[1]} != index dim {INDEX_DIM}. "
                f"Check EMB_MODEL and FAISS index.",
            )
        D, I = await _faiss_search(embs, req.topk)
        result = _build_results(I, D, CORPUS)
        return SearchResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)  