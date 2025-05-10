def ds_intent_extract(
    source_start_code,
    source_sol_code,
    source_lib,
    target_lib,
    target_input,
    target_start_code,
):
    """
    Generates a prompt for extracting the intent behind a given data science library code snippet.

    Args:
        source_start_code (str): Initial code or context provided (e.g., imports, setup).
        source_sol_code (str): The main code snippet or solution code to analyze.
        source_lib (str): The name of the source data science library (e.g., Pandas, NumPy, scikit-learn).
        target_lib (str): The name of the target data science library (not used in this prompt).
        source_input (str): Description or snippet of the source data input context.
        source_output (str): Description or snippet of the output produced by the source code.
        target_input (str): Description or snippet of the target data input context (not used here).
        target_output (str): Description or snippet of the desired target output (not used here).
        target_start_code (str): Initial code or context for the target scenario (not used here).

    Returns:
        str: A formatted prompt for intent extraction.
    """
    return f"""You are a data assistant. Your task is to analyze the given {source_lib} code and extract the intent behind it. The intent should describe the high-level operation or purpose of the code.

**Instructions**:
1. Review the provided code and understand the operation it performs in a data science context.
2. Identify the high-level intent or purpose of the code. Examples include:
   - "Filter rows based on a condition."
   - "Aggregate data by a certain column and compute statistics."
   - "Perform a transformation on one or more columns."
3. Output the identified intent as a single concise sentence. Do not include code, comments, or explanations.

- Source Library: {source_lib}
- Source Start Code:
{source_start_code}

- Source Solution Code:
{source_sol_code}

- Identified Intent:
[YOUR INTENT HERE]
"""


cot_intent_extract = """You are a data assisant. Your task is to analyze the given [[source_library]] library code and extract the intent behind it.

- Source [[source_library]] Query:
[[source_code]]

- Source [[source_library]] Example Input:
[[source_example_input]]

Let's think step by step:
-- Step1: """


def ds_intent_prompt(
    source_start_code,
    source_sol_code,
    source_lib,
    target_lib,
    target_input,
    target_start_code,
    source_question,
    target_question,
):
    return f"""You are a data assistant. Your task is to translate the given {source_lib} library code to {target_lib} library code.

- Source {source_lib} Question:
{source_question}
- Source {source_lib} Start Code:
{source_start_code}
- Source {source_lib} Solution Code:
{source_sol_code}

- Target {target_lib} Question:
{target_question}
- Target {target_lib} Example Input:
{target_input}
- Target {target_lib} Start Code:
{target_start_code}
- Target {target_lib} Solution Code:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_lib} solution code.
```Python
"""
