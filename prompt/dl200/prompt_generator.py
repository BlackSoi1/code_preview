import argparse
import os

from prompt_zero import dl_zero_prompt_complete
from prompt_intent import dl_intent_extract, dl_intent_prompt
from prompt_rag import dl_rag_prompt, dl_rag_api_identify

from util.utils import load_jsonl, write_prompts

# Configuration for different prompt generation versions
# Adjust folder names and functions as needed.
prompt_versions = {
    "zero_shot_complete": {
        "function": dl_zero_prompt_complete,
        "folder": "zero_shot",
    },
    "intent_extract": {
        "function": dl_intent_extract,
        "folder": "intent",
    },
    "intent": {
        "function": dl_intent_prompt,
        "folder": "intent",
    },
    "rag_op_identify": {
        "function": dl_rag_api_identify,
        "folder": "rag",
    },
    "rag": {
        "function": dl_rag_prompt,
        "folder": "rag",
    },
}

# Supported transformations between deep learning frameworks
# For example, PyTorch to TensorFlow and vice versa.
transformations = [
    {"from_framework": "PyTorch", "to_framework": "TensorFlow"},
    {"from_framework": "TensorFlow", "to_framework": "PyTorch"},
]


def decouple_question_schema(datasets):
    """
    Extract and separate framework-specific fields from the dataset.

    Each dataset entry should contain fields for both PyTorch and TensorFlow examples.
    For instance:
      - "question": The overall transformation or intent description.
      - "pytorch_example_input", "pytorch_start_code", "pytorch_sol_code": PyTorch code and examples.
      - "tensorflow_example_input", "tensorflow_start_code", "tensorflow_sol_code": TensorFlow code and examples.

    Args:
        datasets (list): A list of dataset dictionaries.

    Returns:
        tuple: A tuple containing lists for question, PyTorch inputs/code, and TensorFlow inputs/code:
               (question_list,
                pytorch_example_input_list, pytorch_start_code_list, pytorch_sol_code_list,
                tensorflow_example_input_list, tensorflow_start_code_list, tensorflow_sol_code_list)
    """
    question_list = []
    pytorch_example_input_list = []
    pytorch_start_code_list = []
    pytorch_sol_code_list = []
    tensorflow_example_input_list = []
    tensorflow_start_code_list = []
    tensorflow_sol_code_list = []

    for instance in datasets:
        question_list.append(instance["question"])
        pytorch_example_input_list.append(instance["pytorch_example_input"])
        pytorch_start_code_list.append(instance["pytorch_start_code"])
        pytorch_sol_code_list.append(instance["pytorch_sol_code"])
        tensorflow_example_input_list.append(instance["tensorflow_example_input"])
        tensorflow_start_code_list.append(instance["tensorflow_start_code"])
        tensorflow_sol_code_list.append(instance["tensorflow_sol_code"])

    return (
        question_list,
        pytorch_example_input_list,
        pytorch_start_code_list,
        pytorch_sol_code_list,
        tensorflow_example_input_list,
        tensorflow_start_code_list,
        tensorflow_sol_code_list,
    )


def generate_prompts(
    version_func,
    data_list,
    from_start_code,
    from_sol_code,
    from_framework,
    to_framework,
    to_example_input,
    to_start_code,
    from_question,
    to_question,
):
    """
    Generate a list of prompts by applying the version-specific prompt generation function.

    If the prompt is intent-related (dl_intent_prompt), both source and target questions are provided.
    Otherwise, questions are not needed, and we focus on code and inputs.

    Args:
        version_func (function): The prompt generation function.
        data_list (list): List of dataset instances.
        from_start_code (list): Source framework start code snippets.
        from_sol_code (list): Source framework solution code snippets.
        from_framework (str): Name of the source deep learning framework (e.g., "PyTorch").
        to_framework (str): Name of the target deep learning framework (e.g., "TensorFlow").
        to_example_input (list): Target framework example input snippets.
        to_start_code (list): Target framework start code snippets.
        from_question (list): Source framework question/intent descriptions.
        to_question (list): Target framework question/intent descriptions.

    Returns:
        list: A list of generated prompt strings.
    """
    if version_func.__name__ == "dl_intent_prompt":
        # Intent prompts include questions from both source and target frameworks
        return [
            version_func(
                from_start_code[i],
                from_sol_code[i],
                from_framework,
                to_framework,
                to_example_input[i],
                to_start_code[i],
                from_question[i],
            )
            for i in range(len(data_list))
        ]
    else:
        # Other prompts only require code and example inputs
        return [
            version_func(
                from_start_code[i],
                from_sol_code[i],
                from_framework,
                to_framework,
                to_example_input[i],
                to_start_code[i],
            )
            for i in range(len(data_list))
        ]


def process_prompts(data_path, prompt_base):
    """
    Main function to process datasets and generate prompts for each configured version and framework transformation.

    Steps:
    1. Load data from the specified JSONL file.
    2. Extract necessary fields for PyTorch and TensorFlow.
    3. For each version and transformation, generate prompts using the corresponding functions.
    4. Write the generated prompts to output JSONL files in the specified directories.

    Args:
        data_path (str): Path to the input data in JSONL format.
        prompt_base (str): Base directory to store the generated prompt files.
    """
    # Load the dataset
    data_list = load_jsonl(data_path)

    # Extract fields specific to PyTorch and TensorFlow
    (
        question_list,
        pytorch_example_input_list,
        pytorch_start_code_list,
        pytorch_sol_code_list,
        tensorflow_example_input_list,
        tensorflow_start_code_list,
        tensorflow_sol_code_list,
    ) = decouple_question_schema(data_list)

    # Map the framework names to their corresponding extracted lists
    framework_map = {
        "PyTorch": {
            "question": question_list,
            "example_input": pytorch_example_input_list,
            "start_code": pytorch_start_code_list,
            "sol_code": pytorch_sol_code_list,
            "data": data_list,
        },
        "TensorFlow": {
            "question": question_list,  # same question list since it's a universal question field
            "example_input": tensorflow_example_input_list,
            "start_code": tensorflow_start_code_list,
            "sol_code": tensorflow_sol_code_list,
            "data": data_list,
        },
    }

    # Iterate through each prompt version and transformation
    for version_name, version_info in prompt_versions.items():
        version_func = version_info["function"]
        version_folder = version_info["folder"]

        for transformation in transformations:
            from_fw = transformation["from_framework"]
            to_fw = transformation["to_framework"]

            prompts = generate_prompts(
                version_func,
                data_list,
                framework_map[from_fw]["start_code"],
                framework_map[from_fw]["sol_code"],
                from_fw,
                to_fw,
                framework_map[to_fw]["example_input"],
                framework_map[to_fw]["start_code"],
                framework_map[from_fw]["question"],
                framework_map[to_fw]["question"],
            )

            # Construct the output file path
            prompt_path = os.path.join(
                prompt_base,
                version_folder,
                f"{version_name}_{from_fw.lower()}_to_{to_fw.lower()}.jsonl",
            )

            write_prompts(prompts, framework_map[from_fw]["data"], prompt_path)

    print("All prompts generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts for transformations between deep learning frameworks."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data in JSONL format",
    )
    parser.add_argument(
        "--prompt_base",
        type=str,
        required=True,
        help="Base directory to store the generated prompt files",
    )
    args = parser.parse_args()

    process_prompts(args.data_path, args.prompt_base)
