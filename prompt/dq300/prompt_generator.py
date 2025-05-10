import argparse
import os

from prompt_zero import dq_zero_prompt_complete
from prompt_intent import dq_extract_intent, dq_intent_prompt
from prompt_rag import dq_rag_op_identify, dq_rag_prompt

from util.table_schema import (
    generate_schema_prompt_pandas,
    generate_schema_prompt_postgresql,
)

from util.utils import load_jsonl, write_prompts

# Configuration for different prompt generation versions
prompt_versions = {
    "zero_shot_complete": {"function": dq_zero_prompt_complete, "folder": "zero_shot"},
    "intent_extract": {"function": dq_extract_intent, "folder": "intent"},
    "intent": {"function": dq_intent_prompt, "folder": "intent"},
    "rag_op_identify": {"function": dq_rag_op_identify, "folder": "rag"},
    "rag": {"function": dq_rag_prompt, "folder": "rag"},
}

# Supported transformations between database types
transformations = [
    {"from_db": "PostgreSQL", "to_db": "Pandas"},
    {"from_db": "Pandas", "to_db": "PostgreSQL"},
]


def generate_schema_list(datasets, schema_generator):
    """
    Generate a list of schema prompts for each dataset using the provided generator function.

    Args:
        datasets (list): List of dataset dictionaries.
        schema_generator (function): Function that takes a db_path and returns a schema prompt.

    Returns:
        list: List of schema prompts corresponding to each dataset.
    """
    return [schema_generator(data["db_path"]) for data in datasets]


def generate_prompts(
    version_func, data_list, from_list, to_list, from_db, to_db, from_schema, to_schema
):
    """
    Generate a list of prompts by applying the version function to each data instance.

    Args:
        version_func (function): The prompt generation function.
        data_list (list): List of data instances.
        from_list (list): List of source queries (e.g., SQL or Pandas) for each data instance.
        to_list (list): List of target queries (e.g., SQL or Pandas) for each data instance.
        from_db (str): Source database type.
        to_db (str): Target database type.
        from_schema (list): List of schema prompts for the source database.
        to_schema (list): List of schema prompts for the target database.

    Returns:
        list: A list of generated prompt strings.
    """
    return [
        version_func(from_list[i], from_db, to_db, from_schema[i], to_schema[i])
        for i in range(len(data_list))
    ]


def process_prompts(data_path, prompt_base):
    """
    Main entry point for processing prompts. Loads data, generates schemas, and writes prompts
    for all configured transformations and prompt versions.

    Args:
        data_path (str): Path to the input data in JSONL format.
        prompt_base (str): Base directory where generated prompts will be saved.
    """
    # Load data
    data_list = load_jsonl(data_path)

    # Extract SQL and Pandas queries
    postgresql_list = [data["SQL"] for data in data_list]
    pandas_list = [data["pandas_query"] for data in data_list]

    # Generate schema prompts for each dataset
    pandas_schema_list = generate_schema_list(data_list, generate_schema_prompt_pandas)
    postgresql_schema_list = generate_schema_list(
        data_list, generate_schema_prompt_postgresql
    )

    # Prepare lookup maps for both database types
    sql_map = {
        "PostgreSQL": {
            "list": postgresql_list,
            "schema": postgresql_schema_list,
            "data": data_list,
        },
        "Pandas": {
            "list": pandas_list,
            "schema": pandas_schema_list,
            "data": data_list,
        },
    }

    # Generate prompts for each version and transformation
    for version_name, version_info in prompt_versions.items():
        version_func = version_info["function"]
        version_folder = version_info["folder"]
        for transformation in transformations:
            from_db, to_db = transformation["from_db"], transformation["to_db"]

            prompts = generate_prompts(
                version_func,
                sql_map[from_db]["data"],
                sql_map[from_db]["list"],
                sql_map[to_db]["list"],
                from_db,
                to_db,
                sql_map[from_db]["schema"],
                sql_map[to_db]["schema"],
            )

            # Construct the output prompt path
            prompt_path = os.path.join(
                prompt_base,
                version_folder,
                f"{version_name}_{from_db.lower()}_to_{to_db.lower()}.jsonl",
            )

            write_prompts(prompts, sql_map[from_db]["data"], prompt_path)

    print("All prompts generated successfully.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate prompts for SQL and Pandas transformations."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data in JSONL format",
    )
    parser.add_argument(
        "--prompt_base",
        type=str,
        required=True,
        help="Base directory to save generated prompts",
    )
    args = parser.parse_args()

    # Execute the main prompt processing function
    process_prompts(args.data_path, args.prompt_base)
