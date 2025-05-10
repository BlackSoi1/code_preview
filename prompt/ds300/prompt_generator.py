import argparse
import os

from prompt_zero import ds_zero_prompt_complete
from prompt_intent import ds_intent_extract, ds_intent_prompt
from prompt_rag import ds_rag_prompt, ds_rag_api_identify

from util.utils import load_jsonl, write_prompts

# Configuration for different prompt generation versions
prompt_versions = {
    "zero_shot_complete": {
        "function": ds_zero_prompt_complete,
        "folder": "zero_shot",
    },
    "intent_extract": {
        "function": ds_intent_extract,
        "folder": "intent",
    },
    "intent": {
        "function": ds_intent_prompt,
        "folder": "intent",
    },
    "rag_op_identify": {
        "function": ds_rag_api_identify,
        "folder": "rag",
    },
    "rag": {
        "function": ds_rag_prompt,
        "folder": "rag",
    },
}

# Supported transformations between library types (e.g., Pandas to Numpy, Numpy to Pandas)
transformations = [
    {"from_lib": "Pandas", "to_lib": "Numpy"},
    {"from_lib": "Numpy", "to_lib": "Pandas"},
]


def decouple_question_schema(datasets):
    """
    Extract and separate various fields (e.g., questions, code snippets) for Numpy and Pandas from the dataset.

    Args:
        datasets (list): List of dataset dictionaries.

    Returns:
        tuple: A tuple containing lists for Numpy and Pandas related data:
               (numpy_question_list, numpy_example_input_list, numpy_start_code_list, numpy_sol_code_list,
                pandas_question_list, pandas_example_input_list, pandas_start_code_list, pandas_sol_code_list)
    """
    numpy_question_list = []
    numpy_example_input_list = []
    numpy_start_code_list = []
    numpy_sol_code_list = []

    pandas_question_list = []
    pandas_example_input_list = []
    pandas_start_code_list = []
    pandas_sol_code_list = []

    for instance in datasets:
        numpy_question_list.append(instance["numpy_question"])
        numpy_example_input_list.append(instance["numpy_example_input"])
        numpy_start_code_list.append(instance["numpy_start_code"])
        numpy_sol_code_list.append(instance["numpy_sol_code"])

        pandas_question_list.append(instance["pandas_question"])
        pandas_example_input_list.append(instance["pandas_example_input"])
        pandas_start_code_list.append(instance["pandas_start_code"])
        pandas_sol_code_list.append(instance["pandas_sol_code"])

    return (
        numpy_question_list,
        numpy_example_input_list,
        numpy_start_code_list,
        numpy_sol_code_list,
        pandas_question_list,
        pandas_example_input_list,
        pandas_start_code_list,
        pandas_sol_code_list,
    )


def generate_prompts(
    version_func,
    data_list,
    from_start_code,
    from_sol_code,
    from_lib,
    to_lib,
    to_example_input,
    to_start_code,
    from_question,
    to_question,
):
    """
    Generate a list of prompts by applying a version-specific prompt generation function.

    For intent-related prompts (ds_intent_prompt), questions from both source and target libraries are included.
    Otherwise, only source and target code snippets and inputs are needed.

    Args:
        version_func (function): The prompt generation function.
        data_list (list): List of dataset instances.
        from_start_code (list): Source library start code snippets.
        from_sol_code (list): Source library solution code snippets.
        from_lib (str): Name of the source library (e.g., "Numpy" or "Pandas").
        to_lib (str): Name of the target library.
        to_example_input (list): Target library example input snippets.
        to_start_code (list): Target library start code snippets.
        from_question (list): Source library question or intent descriptions.
        to_question (list): Target library question or intent descriptions.

    Returns:
        list: A list of generated prompt strings.
    """
    if version_func.__name__ == "ds_intent_prompt":
        # For intent prompts, also include questions from both source and target
        return [
            version_func(
                from_start_code[i],
                from_sol_code[i],
                from_lib,
                to_lib,
                to_example_input[i],
                to_start_code[i],
                from_question[i],
                to_question[i],
            )
            for i in range(len(data_list))
        ]
    else:
        # For other prompts, the prompt generation does not require the questions
        return [
            version_func(
                from_start_code[i],
                from_sol_code[i],
                from_lib,
                to_lib,
                to_example_input[i],
                to_start_code[i],
            )
            for i in range(len(data_list))
        ]


def process_prompts(data_path, prompt_base):
    """
    Process the dataset to generate prompts for each version and library transformation.

    This function:
    1. Loads the data from a JSONL file.
    2. Extracts fields necessary for prompt generation (Numpy/Pandas questions, code, etc.).
    3. Iterates through each prompt version and supported transformation.
    4. Uses the corresponding prompt generation function to create prompts.
    5. Writes the generated prompts to output JSONL files.

    Args:
        data_path (str): Path to the input data in JSONL format.
        prompt_base (str): Base directory where generated prompts will be saved.
    """
    # Load data
    data_list = load_jsonl(data_path)

    # Extract fields related to Numpy and Pandas
    (
        numpy_question_list,
        numpy_example_input_list,
        numpy_start_code_list,
        numpy_sol_code_list,
        pandas_question_list,
        pandas_example_input_list,
        pandas_start_code_list,
        pandas_sol_code_list,
    ) = decouple_question_schema(data_list)

    # Prepare maps for Numpy and Pandas
    lib_map = {
        "Numpy": {
            "question": numpy_question_list,
            "example_input": numpy_example_input_list,
            "start_code": numpy_start_code_list,
            "sol_code": numpy_sol_code_list,
            "data": data_list,
        },
        "Pandas": {
            "question": pandas_question_list,
            "example_input": pandas_example_input_list,
            "start_code": pandas_start_code_list,
            "sol_code": pandas_sol_code_list,
            "data": data_list,
        },
    }

    # Generate and write prompts for each version and transformation
    for version_name, version_info in prompt_versions.items():
        version_func = version_info["function"]
        version_folder = version_info["folder"]

        for transformation in transformations:
            from_lib = transformation["from_lib"]
            to_lib = transformation["to_lib"]

            prompts = generate_prompts(
                version_func,
                data_list,
                lib_map[from_lib]["start_code"],
                lib_map[from_lib]["sol_code"],
                from_lib,
                to_lib,
                lib_map[to_lib]["example_input"],
                lib_map[to_lib]["start_code"],
                lib_map[from_lib]["question"],
                lib_map[to_lib]["question"],
            )

            # Construct output file path
            prompt_path = os.path.join(
                prompt_base,
                version_folder,
                f"{version_name}_{from_lib.lower()}_to_{to_lib.lower()}.jsonl",
            )

            write_prompts(prompts, lib_map[from_lib]["data"], prompt_path)

    print("All prompts generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts for transformations between data science libraries."
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
