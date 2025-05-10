import json
import os
import argparse

# Paths
metadata_path = "/src/rag/database/guidelines.jsonl"
rag_prompt_base = "/dq300_prompt_rag/rag_translation"

# Load metadata
with open(metadata_path, "r") as f:
    metadata = [json.loads(line) for line in f]


# Function to retrieve operations that exactly match the query
def keyword_search(query, top_k=1):
    """Search metadata for operations that exactly match the query."""
    query_lower = query.strip().lower()

    # Find operations that exactly match the query
    results = []
    for item in metadata:
        operation_name = item["operation"].strip().lower()
        if operation_name == query_lower:
            results.append(
                {
                    "Operation": item["operation"],
                    "PostgreSQL": item["postgreSQL"],
                    "Pandas": item["pandas"],
                }
            )

    # Return top_k results (though there should typically be at most one exact match)
    return results[:top_k]


def process_response(response):
    """Process the response string to remove brackets, split into a list, and remove duplicates."""
    response = response.replace("[", "").replace("]", "")
    operations = [
        op.strip().strip('"') for op in response.split(",")
    ]  # Remove quotes and spaces
    return list(set(operations))  # Remove duplicates


def format_guideline(results_list):
    """Format each list of results into guideline strings."""
    formatted_guidelines = []
    for results in results_list:
        unique_results = {
            result["Operation"]: result for result in results
        }.values()  # Deduplicate
        guideline_string = ""
        for i, result in enumerate(unique_results, 1):
            guideline_string += f"{i}. {result['Operation']}:\nPostgreSQL Solution: {result['PostgreSQL']}\nPandas Solution: {result['Pandas']}\n\n"
        formatted_guidelines.append(guideline_string.strip())  # Remove trailing newline
    return formatted_guidelines


def replace_guideline_in_prompts(input_file, guideline_strings, output_file):
    """Replace the [GUIDELINES] placeholder in the prompts and save the updated file."""
    with open(input_file, "r") as f:
        prompts = [json.loads(line) for line in f]

    # Replace the guideline for each prompt line-by-line
    for i, prompt in enumerate(prompts):
        prompt["prompt"] = prompt["prompt"].replace(
            "[GUIDELINES]", guideline_strings[i]
        )

    # Save the updated prompts
    with open(output_file, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process RAG prompts with keyword retrieval."
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
    args = parser.parse_args()

    # Extract information from input filename
    input_file = args.input_path
    filename = os.path.basename(input_file)
    if "pandas_to_postgresql" in filename:
        direction = "pandas_to_postgresql"
    else:
        direction = "postgresql_to_pandas"

    # Read input JSONL file
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Process each object and retrieve keyword search results
    all_results = []
    for entry in data:
        response = entry[
            "response"
        ]  # Assuming the field 'response' contains the operations
        operations = process_response(response)

        # Retrieve top results for each operation
        results = []
        for operation in operations:
            top_k_results = keyword_search(operation, top_k=1)
            results.extend(top_k_results)

        # Append deduplicated results as a list
        all_results.append(
            list({result["Operation"]: result for result in results}.values())
        )

    # Format all results into guideline strings
    formatted_guidelines = format_guideline(all_results)

    # Select the appropriate prompt file
    original_prompt_file = os.path.join(rag_prompt_base, f"ragv1_{direction}.jsonl")
    output_file = args.output_path

    # Replace [GUIDELINES] in the prompts and save the file
    replace_guideline_in_prompts(
        original_prompt_file, formatted_guidelines, output_file
    )

    print(f"Processed prompts saved to {output_file}")


if __name__ == "__main__":
    main()
