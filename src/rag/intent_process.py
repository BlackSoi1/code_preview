import json
import os
import argparse
from tqdm import tqdm


def replace_intent_in_prompts(input_file, guideline_strings, output_file):
    """Replace the [APIS] placeholder in the prompts and save the updated file."""
    with open(input_file, "r") as f:
        prompts = [json.loads(line) for line in f]

    # Replace the guideline for each prompt line-by-line
    for i, prompt in enumerate(prompts):
        prompt["prompt"] = prompt["prompt"].replace("[INTENT]", guideline_strings[i])

    # Save the updated prompts
    with open(output_file, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process RAG prompts with FAISS retrieval."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input JSONL file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output prompt JSONL file.",
    )
    parser.add_argument(
        "--intent_prompt_base",
        type=str,
        required=True,
        help="Path to intermediate prompt JSONL file.",
    )
    args = parser.parse_args()
    input_file = args.input_path
    filename = os.path.basename(input_file)
    # Read input JSONL file
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    if "pd_to_np" in filename:
        direction = "pd_to_np"
    elif "np_to_pd" in filename:
        direction = "np_to_pd"
    elif "py_to_tf" in filename:
        direction = "py_to_tf"
    elif "tf_to_py" in filename:
        direction = "tf_to_py"
    elif "pandas_to_postgresql" in filename:
        direction = "pandas_to_postgresql"
    else:
        direction = "postgresql_to_pandas"
    # Process each object and retrieve FAISS results with tqdm progress bar
    all_results = []
    for entry in tqdm(data, desc="Processing entries", unit="entry"):
        response = entry[
            "response"
        ]  # Assuming the field 'response' contains the operations

        # Append deduplicated results as a list
        all_results.append(response)

    # Select the appropriate prompt file
    original_prompt_file = os.path.join(
        args.intent_prompt_base, f"extract_intent_{direction}.jsonl"
    )
    output_file = args.output_path

    # Replace [APIs] in the prompts and save the file
    replace_intent_in_prompts(original_prompt_file, all_results, output_file)

    print(f"Processed prompts saved to {output_file}")


if __name__ == "__main__":
    main()
