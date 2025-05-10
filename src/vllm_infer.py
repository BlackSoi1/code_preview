import argparse
import os
import re
import sys
import json
import time
import torch
import jsonlines
from tqdm import tqdm
from vllm import LLM, SamplingParams

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
MAX_INT = sys.maxsize
MAX_TRY = 10
INVALID_ANS = "[INVALID]"


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def load_jsonl(file_path):
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def new_directory(path):
    """
    Create a new directory if it does not already exist.

    Args:
        path (str): Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def write_response(results, data_list, output_path):
    """
    Write generated responses into the provided data structure and save as JSONL.

    Args:
        results (list): A list of generated response strings.
        data_list (list): A list of dictionaries containing the original data.
        output_path (str): Path to the output JSONL file.
    """
    for i, data in enumerate(data_list):
        data["response"] = results[i]

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for instance in data_list:
                f.write(json.dumps(instance) + "\n")


def filter_output(response):
    """
    Filter out invalid responses, based on minimal length criteria.

    Args:
        response (str): The generated response from the model.

    Returns:
        str or None: The original response if valid, otherwise None.
    """
    if len(response.split()) < 1 or len(response) <= 1:
        return None
    return response


def process_batch_data(data_list, batch_size=1):
    """
    Split the input data into batches.

    Args:
        data_list (list): List of items (e.g., prompts) to batch.
        batch_size (int): Number of items per batch.

    Returns:
        list: A list of batches, each a list of items.
    """
    n_batches = len(data_list) // batch_size
    batch_data = []
    for i in range(n_batches - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    # Remaining items in the last batch
    last_start = (n_batches - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])

    return batch_data


def post_process_single_response(response, language="Python"):
    """
    Post-process a single response by extracting code blocks and cleaning them.

    Steps:
    1. Extract code from fenced code blocks.
    2. Remove lines containing '...'.

    Args:
        response (str): The raw model-generated text.
        language (str): The expected language of the code block.

    Returns:
        str: Cleaned code string with unwanted lines removed.
    """
    code_pattern = rf"```(?:{language.lower()}|{language})?\n([\s\S]*?)\n```"
    match = re.search(code_pattern, response, re.IGNORECASE)

    if match:
        code = match.group(1)
    else:
        # If the regex fails, attempt manual clean-up
        code = response.strip()
        code = (
            code.replace("```", "")
            .replace("sql", "")
            .replace("SQL", "")
            .replace("Python", "")
            .replace("python", "")
        )

    # Remove lines containing '...'
    cleaned_lines = [line for line in code.split("\n") if "..." not in line]
    cleaned_code = "\n".join(cleaned_lines).strip()

    return cleaned_code


def post_process_responses(responses, language):
    """
    Post-process a list of responses.

    Args:
        responses (list): List of raw model responses.
        language (str): Target language for code extraction and cleaning.

    Returns:
        list: A list of cleaned responses.
    """
    return [post_process_single_response(response, language) for response in responses]


# -------------------------------------------------------------------------
# Inference Functions
# -------------------------------------------------------------------------
def infer_batch(model_path, batch_size, input_file, language):
    """
    Perform batched inference using a specified model and prompt file.

    Args:
        model_path (str): Path to the model.
        batch_size (int): Batch size for inference.
        input_file (str): Path to the JSONL file containing prompts.
        language (str): Target language for code post-processing.

    Returns:
        list: A list of generated responses.
    """
    print(f"Input file: {input_file}", flush=True)

    # Load input data
    data = []
    prompts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            prompts.append(item["prompt"])
            data.append(item)

    # Prepare batches
    start, end = 0, MAX_INT
    prompts = prompts[start:end]
    data = data[start:end]

    batch_prompts = process_batch_data(prompts, batch_size=batch_size)
    batch_data = process_batch_data(data, batch_size=batch_size)
    print("Number of samples to infer:", len(prompts))

    # Initialize LLM based on the model_path
    if "Qwen2.5-72B" in model_path or "Meta-Llama-3.1-70B" in model_path:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            max_model_len=8192,
            gpu_memory_utilization=0.85,
        )
    else:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
        )

    tokenizer = llm.get_tokenizer()
    stop_tokens = ["</FINAL_ANSWER>", "<|EOT|>"]

    # Define sampling parameters
    if "Meta-Llama-3" in model_path:
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        sampling_params = SamplingParams(
            temperature=0.001,
            top_k=10,
            top_p=0.99,
            max_tokens=3000,
            stop=stop_tokens,
            stop_token_ids=stop_token_ids,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.001,
            top_k=10,
            top_p=0.95,
            max_tokens=3000,
            stop=stop_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )

    print("Sampling Params:", sampling_params)

    results = []
    # Perform inference batch-by-batch
    for idx, (batch_prompt, batch_sample) in enumerate(
        tqdm(zip(batch_prompts, batch_data), total=len(batch_prompts))
    ):
        print(f"Inferencing {idx}th batch...", flush=True)
        # Ensure batch_prompt is a list
        if not isinstance(batch_prompt, list):
            batch_prompt = [batch_prompt]

        # Construct conversation format
        conversations = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch_prompt
        ]

        if idx == 0:
            print("Conversation Example:", conversations[0])

        # Retry logic for invalid responses
        have_invalid = True
        attempt_count = 0
        while have_invalid and attempt_count < MAX_TRY:
            attempt_count += 1
            with torch.no_grad():
                completions = llm.generate(conversations, sampling_params)

            # Check validity of outputs
            for output in completions:
                gen_seq = output.outputs[0].text
                filter_seq = filter_output(gen_seq)
                processed_gen_seq = post_process_single_response(gen_seq, language)
                print(f"Processed Response: {processed_gen_seq}", flush=True)

                if filter_seq is None:
                    have_invalid = True
                    break
                else:
                    have_invalid = False

            if attempt_count >= MAX_TRY and have_invalid:
                print("********** Invalid Output After MAX_TRY Attempts **********")
                # Mark output as invalid if still invalid
                have_invalid = False

        # Store results
        for output, sample in zip(completions, batch_sample):
            gen_seq = output.outputs[0].text
            filter_seq = filter_output(gen_seq)
            if filter_seq is None:
                gen_seq = INVALID_ANS
            results.append(gen_seq)

    return results


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to the input JSONL file containing prompts.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path for the output JSONL file."
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for inference."
    )
    parser.add_argument("--gpu", type=str, default="0", help="Which GPU to use.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Output path: {args.output_path}")

    # Determine target language
    file_basename = os.path.basename(args.prompt_path)
    if file_basename.split(".")[0].split("_")[-1] == "postgresql":
        target_language = "SQL"
    elif "identify" in file_basename:
        target_language = "JSON"
    else:
        target_language = "Python"

    # Run inference
    results = infer_batch(
        args.model_path, args.batch_size, args.prompt_path, target_language
    )

    # Post-process results
    results = post_process_responses(results, target_language)

    # Load original data and write responses
    data_list = load_jsonl(args.prompt_path)
    write_response(results, data_list, args.output_path)
