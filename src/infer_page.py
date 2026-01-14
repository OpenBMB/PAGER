import argparse
import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from copy import deepcopy
import torch
torch.cuda.manual_seed(66)     
torch.cuda.manual_seed_all(66) 

def load_questions_from_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def build_prompt(item):
    question = item.get("question", "")
    page = item.get("page_list", [])[-1]
    prompt = """Page:\n{page}\n
The User asks a question, and the Assistant solves it.
The system will provide the Assistant with a page containing information relevant to answering the question. The assistant should answer the question by combining the Page with its internal knowledge.
When the page provides enough knowledge to answer the question, the assistant should strictly follow the knowledge and writing style from the page. When the page does not provide enough knowledge, the assistant should combine its internal knowledge to answer.
All answers should be as comprehensive and accurate as possible. The assistant should first think through the reasoning process, then provide the precise and short final answer. 
The output format for the final answer should be enclosed within <answer></answer> tags. You need to first present the reasoning process, then give the final answer, like: “Reasoning process here\n\n<answer> Only the short final answer here </answer>”.
\n\nUser:{question}\nAssistant:"""

    return prompt.format(page=page, question=question)


def _fit_prompt_and_params(tokenizer, max_model_len,text, base_params):
    """单条输入安全检查，返回安全文本和独立SamplingParams"""
    input_ids = tokenizer.encode(text)
    in_len = len(input_ids)

    # 输入过长 → 截断（保留最后部分）
    if in_len >= max_model_len:
        input_ids = input_ids[-(max_model_len - 1):]
        in_len = len(input_ids)

    # 剩余可用token数
    available_output = max_model_len - in_len
    sp = deepcopy(base_params)
    if sp.max_tokens > available_output:
        sp.max_tokens = available_output
        print(f"----------------- max_tokens is too long ----------------------")

    safe_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    return safe_text, sp
    
def main():

    parser = argparse.ArgumentParser(
        description="Batch inference using vLLM for offline batch processing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yyk/yyk08/qwq32b",
        help="Path to the model or model identifier",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/yyk/yyk08/CacheNote_HX/0927_new_setting/cite_page/out_page/qwen32b/cite_2wiki_500_delete.jsonl",
        help="Path to JSON file containing a list of dicts with keys 'query', 'passage', 'page_plan'",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/lixinze/webwalker/R1-Searcher-main/reproduction/DeepNote/output/processed/infer/infer_bamboogle_qwq.jsonl",
        help="If set, write all (input, output) pairs as JSON Lines to this file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU IDs to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=66,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("Arguments:", args)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    # Load questions
    data_list = load_questions_from_file(args.input_file)
    print(f"Loaded {len(data_list)} examples from {args.input_file}")
    
    
        # Initialize the vLLM model
    print(f"Loading model: {args.model}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

    llm = LLM(
        model=args.model,
        tensor_parallel_size=len(args.gpu_ids.split(",")),
        trust_remote_code=True,
        seed=args.seed,
    )

    print("Model loaded successfully")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.02,
        seed=args.seed )
    

    # Prepare all prompts
    prompts = []
    per_params = []
    for item in data_list:
        prompt = build_prompt(item)
        base_params = deepcopy(sampling_params)
        safe_text, sp = _fit_prompt_and_params(tokenizer, 40900,prompt, base_params)
        per_params.append(sp)
        messages = [{"role": "user", "content": safe_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)
        
    print(prompts[0])

    outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch = prompts[i : i + args.batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        outputs.extend(batch_outputs)

    # Process results
    for i, output in enumerate(outputs):
        try:
            result = output.outputs[0].text
            if "<answer>" in result:
                pred_ans = result.split("<answer>")[-1].split("</answer>")[0].strip()
            else:
                pred_ans = "<None>"
        except Exception as e:
            result = f"Error: {e}"
            pred_ans = "<None>"

        data_list[i]["gen_text"] = result
        data_list[i]["pred_ans"] = pred_ans

    print(f"Processing completed for {len(outputs)} examples")

    # Write results to output file
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as fout:
            for rec in data_list:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()