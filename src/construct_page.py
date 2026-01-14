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
        args,
        retrieval_url: str,
        batch_size: int = 8,
        seed=42,

    ):
        """Initialize with LLM, tokenizer, and retrieval service endpoint."""
        self.llm = llm
        self.tokenizer = tokenizer
        self.args = args
        self.retrieval_url = retrieval_url
        self.batch_size = batch_size
        self.max_model_len = 40900
        self.sampling_params = SamplingParams(
            max_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.02,
            seed=seed,  # Use a fixed seed for reproducibility
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

    def retrieve_docs_batch(self, questions: List[str], topk: int) -> List[List[str]]:
        """Retrieve relevant documents for a batch of questions."""
        
        def make_qwen_ret_prompt(query):
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            inst = f'Instruct: {task}\nQuery:{query}'
            return inst
        
        all_doc_lists = []
        all_id_list = []
        # Process questions in smaller chunks if needed to avoid overloading the retrieval service
        chunk_size = (
            1  # You might need to adjust this based on the retrieval service capacity
        )
        for i in range(0, len(questions), chunk_size):
            chunk_questions = questions[i : i + chunk_size]
            chunk_questions = [make_qwen_ret_prompt(q) for q in chunk_questions]
            
            response = requests.post(
                f"{self.retrieval_url}/search", json={"model": "qwen3-emb", "queries": chunk_questions, "topk": topk})
            result = response.json()
            chunk_doc_lists = []
            chunk_id_lists = []
            
            answers = result['contents']
            for answer in answers:
                doc_list = []
                for i in range(len(answer)):
                    retrieve_doc = answer[i]["text"]
                    doc_list.append(retrieve_doc)
                d_ids = [ans['id'] for ans in answer]
                chunk_doc_lists.append(doc_list)
                chunk_id_lists.append(d_ids)
                
            all_doc_lists.extend(chunk_doc_lists)
            all_id_list.extend(chunk_id_lists)

            # Add delay to prevent rate limiting if needed
            if i + chunk_size < len(questions):
                time.sleep(0.5)

        return all_doc_lists,all_id_list
    
    

    def generate_sub_query_batch(
        self, questions: List[str], pages: List[str]
    ) -> tuple[List[List[str]], List[str]]:
        """Generate sub-queries for a batch of pages and retrieve relevant documents."""
        sub_question_prompt_template = """You are a professional question-generation expert. 
First, following the order of the page sections, locate and identify the first unfinished section. An unfinished section is defined as one that contains the special symbol <TO BE FILLED> under the section title.
Please strictly follow the steps and format below to formulate a precise retrieval question for the first unfinished section of the current page, in order to obtain the necessary information to complete that section.


Input Parameters:
- Original question: {question}
- Current page content: {page}

Task Steps:

1. **Parse the page plan:**
    - Read the Page and locate the title and topic of the first section that remains unfilled.

2. **Align with the original question:**
    - Align the section's topic with the core needs of original question to ensure the retrieval question directly serves the original question.

3. **Design the retrieval question. The question must:**
    - Be focused: target only the core subtopic of that section;
    - Be clear: use search terms that can be directly used in a search engine or knowledge base;
    - Be concise: include no unnecessary background.

4. **Output format:**
    - Output only one line containing the English retrieval question, with no additional comments or explanations."""

        # Prepare prompts for batch generation
        prompts = [
            sub_question_prompt_template.format(question=q, page=page)
            for q, page in zip(questions, pages)
        ]
        sub_questions = self.generate_batch(prompts)

        # Clean up sub-questions
        cleaned_sub_questions = []
        for sq in sub_questions:
            cleaned_sub_questions.append(sq)

        # Retrieve documents for each sub-question
        all_doc_lists = []
        valid_sub_questions = []
        all_id_lists = []

        for sq in cleaned_sub_questions:
            max_tries = 5
            tries = 0
            doc_list = []

            # Try to retrieve docs up to max_tries times
            while tries < max_tries and not doc_list:
                # We process sub-questions individually as they may require different contexts
                try:
                    doc_list, id_list = self.retrieve_docs_batch([sq], 5)
                    doc_list = doc_list[0][:5]
                    id_list =  id_list[0][:5]
                except Exception as e:
                    print(f"Retrieval error: {e}. Retrying...")
                tries += 1

            all_doc_lists.append(doc_list)
            valid_sub_questions.append(sq)
            all_id_lists.append(id_list)

        return all_doc_lists, valid_sub_questions,all_id_lists

    def generate_page_batch(
        self,
        questions: List[str],
        pages: List[str],
        sub_questions: List[str],
        doc_lists: List[List[str]],
        id_lists: List[List[str]],
    ) -> List[str]:
        """Generate page content for multiple questions in a batch."""
        
        gen_prompt_template = """You are a professional page content writer.
Please strictly follow the instructions below.
First, following the order of the page sections, locate and identify the first unfinished section. An unfinished section is defined as one that contains the special symbol <TO BE FILLED> under the section title.
Use the retrieved documents, and your internal knowledge to complete the first unfinished section of the page, and generate the page with that section filled in.
Please note that you should only fill in one unfinished section, and that section must be the first unfinished section on the page. Do not fill in additional sections or fill across different sections.

Input:  
- Original question: {question}
- Sub-question for retrieval: {sub_question}
- Retrieved documents: {docs_text}
- Current page: {page}

Task Steps:

1. **Section Completion**  
- Find the first unfinished section in the page, where the section title contains a special placeholder <TO BE FILLED>.
- Using information from the retrieved documents and your internal knowledge, write a short paragraph under that section. The paragraph should:  
    - Be tightly related to the original question.  
    - Focus strictly on the topic of that section.  
    - Avoid redundant or irrelevant information.  
    - Remove the <TO BE FILLED> placeholder under that section.  
    - Retain <TO BE FILLED> placeholders for all other unfinishe sections.  
    - If you have filled the last unfinished section of the page, ensure that no <TO BE FILLED> placeholders remain in the page.
   
2. **Output format**
   - Only output the entire page with the first unfinished section fully filled in. Do not include any comments, explanations, or isolated section content.
   - Your output must be the entire updated page, including the newly filled content, seamlessly integrated into the original page structure.
   - Do not output just the section you filled in—your output must be the entire page, including all content, both existing and newly added.
"""
        prompts = []
        for q, page, sq, docs, ids in zip(
            questions, pages, sub_questions, doc_lists,id_lists
        ):

            wrapped_docs = []
            for d_id, d_content in zip(ids, docs):
                wrapped_docs.append(f"[id: {d_id}]: {d_content}")
            docs_text = "\n".join(wrapped_docs)
            prompt = gen_prompt_template.format(
                question=q, sub_question=sq, docs_text=docs_text, page=page
            )
            prompts.append(prompt)
        results = self.generate_batch(prompts)

        # Clean up generated pages
        cleaned_pages = []
        for page in results:
            cleaned_pages.append(page)

        return cleaned_pages


    def process_batch(self, batch_items: List[Dict], max_iters: int) -> List[Dict]:
        """Process a batch of items through the full pipeline."""
        # Extract questions and page plans
        questions = [item["question"] for item in batch_items]
        initial_page = [item["init_page"] for item in batch_items]

        for idx, item in enumerate(batch_items):
            # item["init_page"] = initial_page[idx]
            item["doc_list"] = []
            item["page_list"] = []
            item["subquestion_list"] = []
            item["doc_id_list"] = []

        # Initialize current pages
        # current_pages = ["null"] * len(batch_items)
        current_pages = initial_page

        # Process iterations
        for iter_idx in range(max_iters):
            print(f"Iteration {iter_idx+1}/{max_iters}")

            # Get items that still need processing (have "<TO BE FILLED>" in their pages)
            active_indices = []
            active_questions = []
            active_current_pages = []

            for idx, page in enumerate(current_pages):
                if page == "null" or "TO BE FILLED".lower() in page.lower():
                    active_indices.append(idx)
                    active_questions.append(questions[idx])
                    active_current_pages.append(page)

            # If no active items, we're done
            if not active_indices:
                print("All pages completed!")
                break

            print(f"Processing {len(active_indices)} active items")

            # Generate sub-queries and retrieve docs
            doc_lists, sub_questions, id_lists = self.generate_sub_query_batch(
                active_questions, active_current_pages
            )

            # Generate new pages
            new_pages = self.generate_page_batch(
                active_questions,
                active_current_pages,
                sub_questions,
                doc_lists,
                id_lists,
            )

            # Update results for active items
            for i, active_idx in enumerate(active_indices):
                batch_items[active_idx]["doc_list"].append(doc_lists[i])
                batch_items[active_idx]["page_list"].append(new_pages[i])
                batch_items[active_idx]["subquestion_list"].append(sub_questions[i])
                batch_items[active_idx]["doc_id_list"].append(id_lists[i])
                current_pages[active_idx] = new_pages[i]

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
        "--retrieval_url",
        type=str,
        default=None,
        help="URL for the retrieval service",
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
    print(args)
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
        args = args,
        retrieval_url=args.retrieval_url,
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