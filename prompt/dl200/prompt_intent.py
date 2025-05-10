def dl_intent_extract(
    source_start_code,
    source_sol_code,
    source_framework,
    target_framework,
    target_input,
    target_start_code,
):
    """
    Generates a prompt for extracting the intent behind a given deep learning framework code snippet.

    Args:
        source_start_code (str): Initial code or context provided (e.g., imports, setup).
        source_sol_code (str): The main code snippet or solution code to analyze.
        source_framework (str): The name of the source deep learning framework (e.g., PyTorch, TensorFlow).
        target_framework (str): The name of the target deep learning framework (not used in this prompt).
        target_input (str): Description of the target data input context (not used here).
        target_start_code (str): Initial code or context for the target scenario (not used here).

    Returns:
        str: A formatted prompt for intent extraction.
    """
    return f"""You are a deep learning engineer. Your task is to analyze the given {source_framework} code and determine its intent. The intent should describe the high-level operation or purpose of the code in a deep learning context.

**Instructions**:
1. Examine the provided code snippet and understand what it accomplishes within a deep learning workflow.
2. Identify the high-level purpose of the code. Some examples might include:
   - "Define and compile a neural network model."
   - "Preprocess training data before feeding it to a model."
   - "Train a model using a specific optimizer and loss function."
   - "Evaluate a trained model on a validation or test dataset."
   - "Perform inference with a trained model to make predictions."
3. Provide the identified intent as a single concise sentence. Do not include code, comments, or explanations.

- Source Deep Learning Framework: {source_framework}

- Source Start Code:
{source_start_code}

- Source Solution Code:
{source_sol_code}

- Identified Intent:
[YOUR INTENT HERE]
"""


cot_intent_extract = """You are a data assisant. Your task is to analyze the given [[source_library]] framework code and extract the intent behind it.

- Source [[source_library]] Query:
[[source_code]]

- Source [[source_library]] Example Input:
[[source_example_input]]

Let's think step by step:
-- Step1: """


def dl_intent_prompt(
    source_start_code,
    source_sol_code,
    source_framework,
    target_framework,
    target_input,
    target_start_code,
    question,
):
    return f"""You are a deep learning engineer. Your task is to translate the given {source_framework} framework code to {target_framework} framework code.

- Question:
{question}

- Source {source_framework} Start Code:
{source_start_code}
- Source {source_framework} Solution Code:
{source_sol_code}

- Target {target_framework} Input Example:
{target_input}
- Target {target_framework} Start Code:
{target_start_code}
- Target {target_framework} Solution Code:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_framework} solution code.
```Python
"""
