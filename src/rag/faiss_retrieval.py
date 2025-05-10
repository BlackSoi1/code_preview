import json
import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# Paths
stella_model_path = "model/stella_en_400M_v5"
faiss_index_path = "PATH/faiss_api"
rag_prompt_base = "ds300_prompt_rag/rag_translation"
model_kwargs = {"device": "cuda", "trust_remote_code": True}
embeddings = HuggingFaceEmbeddings(
    model_name=stella_model_path, model_kwargs=model_kwargs
)

# Load FAISS index and metadata
vector_store = FAISS.load_local(
    faiss_index_path, embeddings, allow_dangerous_deserialization=True
)


# Function to retrieve the most similar operations
def search(query, library, top_k=1):
    """Search FAISS index for top-k similar operations."""
    results = vector_store.similarity_search(
        query,
        k=top_k,
        filter={"Library": library},
    )
    # Retrieve metadata for the top results
    return_res = []
    for idx, res in enumerate(results):
        if idx != -1:  # Ensure valid index
            meta_data = res.metadata
            api_name = meta_data["API_Name"]
            doc_string = meta_data["Docstring"]
            return_res.append(
                {
                    "API Name": api_name,
                    "Doc String": doc_string,
                }
            )

    return return_res


def process_response(response):
    """Process the response string to remove brackets, split into a list, and remove duplicates."""
    response = response.replace("[", "").replace("]", "")
    operations = [
        op.strip().strip('"') for op in response.split(",")
    ]  # Remove quotes and spaces
    return list(set(operations))  # Remove duplicates


def format_doc_strings(results_list):
    """Format each list of results into guideline strings."""
    formatted_guidelines = []
    for results in results_list:
        unique_results = {
            result["API Name"]: result for result in results
        }.values()  # Deduplicate
        guideline_string = ""
        for i, result in enumerate(unique_results, 1):
            guideline_string += f"{i}.{result['API Name']}:\n {result['Doc String']}"
        formatted_guidelines.append(guideline_string.strip())  # Remove trailing newline
    return formatted_guidelines


def replace_apis_in_prompts(input_file, guideline_strings, output_file):
    """Replace the [APIS] placeholder in the prompts and save the updated file."""
    with open(input_file, "r") as f:
        prompts = [json.loads(line) for line in f]

    # Replace the guideline for each prompt line-by-line
    for i, prompt in enumerate(prompts):
        prompt["prompt"] = prompt["prompt"].replace("[APIs]", guideline_strings[i])

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
        "--rag_translation_base",
        type=str,
        required=True,
        help="Path to intermediate prompt JSONL file.",
    )
    args = parser.parse_args()

    # Extract information from input filename
    input_file = args.input_path
    filename = os.path.basename(input_file)
    library = None
    if "pd_to_np" in filename:
        direction = "pd_to_np"
        library = "NumPy"
    elif "np_to_pd" in filename:
        direction = "np_to_pd"
        library = "Pandas"
    elif "py_to_tf" in filename:
        direction = "py_to_tf"
        library = "TensorFlow"
    else:
        direction = "tf_to_py"
        library = "PyTorch"

    # Read input JSONL file
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Process each object and retrieve FAISS results with tqdm progress bar
    all_results = []
    for entry in tqdm(data, desc="Processing entries", unit="entry"):
        response = entry[
            "response"
        ]  # Assuming the field 'response' contains the operations
        api_names = process_response(response)

        # Retrieve top 3 results for each operation
        results = []
        for operation in api_names:
            top_k_results = search(operation, library, top_k=1)
            results.extend(top_k_results)

        # Append deduplicated results as a list
        all_results.append(
            list({result["API Name"]: result for result in results}.values())
        )

    # Format all results into doc strings
    formatted_doc_strings = format_doc_strings(all_results)

    # Select the appropriate prompt file
    original_prompt_file = os.path.join(
        args.rag_translation_base, f"ragv1_{direction}.jsonl"
    )
    output_file = args.output_path

    # Replace [APIs] in the prompts and save the file
    replace_apis_in_prompts(original_prompt_file, formatted_doc_strings, output_file)

    print(f"Processed prompts saved to {output_file}")


if __name__ == "__main__":
    main()
