import json
import os


def load_jsonl(file_path):
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the input JSONL file.

    Returns:
        list: A list of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def create_directory(path):
    """
    Create a directory if it does not already exist.

    Args:
        path (str): Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def write_prompts(prompts, data_list, prompt_path):
    """
    Write the generated prompts into a JSONL file, preserving the original data structure.

    Args:
        prompts (list): A list of generated prompt strings.
        data_list (list): The original dataset entries corresponding to these prompts.
        prompt_path (str): The output JSONL file path.
    """
    if prompt_path:
        directory_path = os.path.dirname(prompt_path)
        create_directory(directory_path)
        with open(prompt_path, "w", encoding="utf-8") as f:
            for i, instance in enumerate(data_list):
                instance["prompt"] = prompts[i]
                f.write(json.dumps(instance) + "\n")
