# Revealing the Attention Floating Mechanism in Masked Diffusion Models

# 📖 Introduction

**PAGER** is a page-driven autonomous knowledge representation framework designed for organizing and utilizing knowledge in Retrieval-Augmented Generation (RAG) scenarios. Specifically, **PAGER** first prompts an LLM to construct a structured cognitive outline for a given query, consisting of multiple slots, each representing a distinct knowledge dimension. Guided by these slots, **PAGER** then iteratively retrieves and refines relevant documents, populating each slot with pertinent information, and ultimately constructs a coherent "knowledge page" that serves as contextual input for answer generation. Experimental results on multiple knowledge-intensive benchmark tasks and across various backbone models show that **PAGER** consistently outperforms all RAG baselines across evaluation metrics.

<img src="figs/method.pdf" width="90%">


# ⚙️ Setup

```bash
conda create --name PAGER python==3.11
conda activate PAGER
git clone https://github.com/OpenBMB/PAGER.git
cd PAGER
pip install -r requirement.txt
```

# Reproduction Guide

This section provides a step-by-step guide to reproduce the LISRec results.

## 1) Evaluation Dataset Download
You can download the evaluation dataset required for this experiment from here. We use the Wiki dataset provided by FlashRAG as the retrieval corpus, and the downloading and processing procedures are detailed [here](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/process-wiki.md).

For the downloaded evaluation dataset and the processed corpus, you need to move them to the `evaluation_dataset` folder.

## 2) Deploying a Retrieval Model Service


### 2.1. Encode the Wikipedia Corpus:
For the downloaded corpus, you need to encode it using the Qwen3-0.6 embedding model and store the resulting embeddings `wiki.npy` in the `embedding` directory. You need to run the following script.

```bash
bash bash/embed.sh
```

### 2.2. Build an Index for the Encoded Embeddings:
For the encoded Wikipedia embeddings, `wiki.npy`, you need to build a FAISS index for it to be used in subsequent retrieval. The constructed index should be saved as `wiki.index` in the `embedding` directory. You need to run the following script.
```bash
bash bash/index.sh
```

### 2.3. Build an Index for the Encoded Embeddings:
Next, you need to run the `vllm_emb.sh` script to deploy the `Qwen3-Embedding-0.6B` model to the GPU and run it in the background.
```bash
bash bash/vllm_emb.sh
```
### 2.4. Deploy the Retrieval Service:
Afterward, you need to run the `ret_serve.sh` script to deploy the retrieval service in the background.
```bash
bash bash/ret_serve.sh
```

## 3) Construct the Structured Page
### 3.1. Construct the Outline:
Construct an outline for the questions in the evaluation dataset. You need to run the following script and store the generated outline in the `output_data/outline` directory.

```bash
bash bash/construct_outline.sh
```
Afterward, extract the structured outline from the outline to serve as the initialized page and store it in the `output_data/outline` directory.
```bash
bash bash/extract_outline.sh
```
### 3.2. Construct the Page:
After obtaining the initialized page, you need to iteratively fill it with knowledge until the final knowledge representations are produced, and store the results in the `output_data/page` directory.
```bash
bash bash/construct_page.sh
```

## 4) Page Inference
### 4.1. Infer the Answers.
After obtaining the pages generated in the `output_data/page` directory, you need to use the generated pages to answer the given questions and store the resulting answers in the `output_data/infer` directory.

```bash
bash bash/infer_page.sh
```
### 4.2. Evaluate the Accuracy of the Answers.
Finally, you can run the following script to evaluate the accuracy of the answers.
```bash
bash bash/evaluate_infer.sh
```

## Contact
If you have questions, suggestions, and bug reports, please email:
```
lxzlxz0716@gmail.com
```
