import argparse
import json
import os
import time
import requests
from tqdm import tqdm
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import ray
from copy import deepcopy
import torch


torch.cuda.manual_seed(66)     
torch.cuda.manual_seed_all(66) 
print("-----------")
class DirectBatchPageGenerator:
    """
    A client for generating pages using direct vLLM batch inference:
    1. Planner: creates overviews/outlines for pages.
    2. Generator: generates full pages based on retrieved docs and overviews.
    """

    def __init__(
        self,
        llm,
        tokenizer,
        batch_size: int = 8,
        seed=66,
    ):
        """Initialize with LLM, tokenizer, and retrieval service endpoint."""
        self.llm = llm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_model_len = 40900
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.02,
            seed=seed,  
        )
        
        
    def _fit_prompt_and_params(self, text: str, base_params: SamplingParams):
        input_ids = self.tokenizer.encode(text)
        in_len = len(input_ids)

        if in_len >= self.max_model_len:
            input_ids = input_ids[-(self.max_model_len - 1):]
            in_len = len(input_ids)

        available_output = self.max_model_len - in_len
        sp = deepcopy(base_params)
        if sp.max_tokens > available_output:
            sp.max_tokens = available_output
            print(f"----------------- max_tokens is too long ----------------------")

        safe_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        return safe_text, sp

    def generate_batch(self, prompts: List[str]) -> List[str]:
    
        """Generate text for a batch of prompts using vLLM."""
        inputs = []
        per_params = []
        for prompt in prompts:
            base_params = deepcopy(self.sampling_params)
            safe_text, sp = self._fit_prompt_and_params(prompt, base_params)
            per_params.append(sp)
            
            messages = [{"role": "user", "content": safe_text}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs.append(text)

        outputs = self.llm.generate(inputs, sampling_params=per_params)
        results = [output.outputs[0].text for output in outputs]
        return results

    def initial_page_batch(
        self,
        questions: List[str],
    ) -> List[str]:

        init_page_prompt_template = """You are a Page Planning Expert.
For a given question, you are tasked with generating a page outline based on the theme of the question.
The outline you generate will serve as the foundation for the subsequent process, during which the outline will be filled with content to create a complete page that assists the reader in answering the given question.
Please strictly follow the instructions below.

Input:  
- Question: {question}

Task Steps:  

1. **Reasoning Analysis**
- First, analyze the theme of the question and thoroughly understand the knowledge required to answer it, as well as the logical relationships between this knowledge. 
- Based on the theme of the question, generate a page outline and identify the key content that should be covered in each section. 
- For each section, propose a suitable and appropriate title, ensuring that each title effectively guides the reader to progressively deepen their understanding of the various aspects of the question. 

1. **Outline Initialization**  
- Generate the outline:  
    - Use # [Main Title] for the main title. The main title is a concise and comprehensive abstraction of the page content.
    - Use ## [Section Title] for each section title. The section title is the section heading (do not reveal the final answer).
    - Insert special marker <TO BE FILLED> under all section titles. 
Make sure the sections of the page are ordered logically, building up the reader's understanding toward answering the question.
The number of sections on the page should be limited to the scope necessary to answer the question, avoiding any overlap of content between sections.

2. **Output**  
- First, you should output the reasoning analysis for initializing the outline. 
- Then, you should generate a special symbol <OUTLINE>, followed by the generated page outline. Note that the content after <OUTLINE> should only include the final generated page outline, without any comments or explanations.
"""

        prompts = [
            init_page_prompt_template.format(question=q)
            for q in questions
        ]
        results = self.generate_batch(prompts)

        cleaned_plans = []
        for plan in results:
            if "</think>" in plan:
                plan = plan.split("</think>")[-1].strip()
            cleaned_plans.append(plan)
        return cleaned_plans

    def process_batch(self, batch_items: List[Dict], max_iters: int) -> List[Dict]:
        """Process a batch of items through the full pipeline."""
        # Extract questions and page plans
        questions = [item["question"] for item in batch_items]
        initial_page = self.initial_page_batch(questions)

        # Initialize fields for storing results
        for idx, item in enumerate(batch_items):
            item["init_page"] = initial_page[idx]
        return batch_items


def load_questions_from_file(input_file: str) -> list:
    """Load JSON objects line by line from a file and return a list."""
    data_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))
    return data_list


def save_json(data_list: list, out_file: str):
    """Write JSON objects line by line to a specified file."""
    with open(out_file, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Direct batch vLLM page generator")

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Path to the model or model identifier",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU IDs to use",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file (JSONL format)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Path to output file (JSONL format)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=10,
        help="Maximum number of iterations for each record",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Number of records to process in each batch",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Process only N records, default: None is all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=66,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

    print("---------Loading Model----------")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=len(args.gpu_ids.split(",")),
        trust_remote_code=True,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    print("Model loaded successfully")

    # Load questions to process
    data_list = load_questions_from_file(args.input_file)
    if args.sample_limit is not None:
        data_list = data_list[: args.sample_limit]
    print(f"Total of {len(data_list)} records to process")

    # Initialize DirectBatchPageGenerator
    page_generator = DirectBatchPageGenerator(
        llm=llm,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Process in batches and write output
    with open(args.out_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(data_list), args.batch_size)):
            batch = data_list[i : i + args.batch_size]
            print(
                f"Processing batch {i//args.batch_size + 1}/{(len(data_list) + args.batch_size - 1)//args.batch_size}"
            )

            try:
                # Process the batch
                processed_batch = page_generator.process_batch(batch, args.max_iters)

                # Write processed items to output file
                for item in processed_batch:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()  # Ensure data is written even if process is interrupted
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Write whatever we have for this batch
                for item in batch:
                    if "doc_list" not in item:
                        item["doc_list"] = []
                    if "page_list" not in item:
                        item["page_list"] = []
                    if "subquestion_list" not in item:
                        item["subquestion_list"] = []
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
    print(f"All processing completed, results written to: {args.out_file}")
    
if __name__ == "__main__":
    main()