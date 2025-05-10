def ds_rag_api_identify(
    source_start_code,
    source_sol_code,
    source_lib,
    target_lib,
    target_input,
    target_start_code,
):
    """
    Generates a prompt for identifying APIs needed for translating code between libraries.

    Args:
        source_start_code (str): The initial input code in the source library.
        source_sol_code (str): The solution code in the source library.
        source_lib (str): The source library name (e.g., Pandas, NumPy).
        target_lib (str): The target library name (e.g., Pandas, NumPy).
        source_input (str): Example input for the source library.
        source_output (str): Example output for the source library.
        target_input (str): Example input for the target library.
        target_output (str): Example output for the target library.
        target_start_code (str): Example starting code for the target library.

    Returns:
        str: A formatted prompt.
    """
    # Example for translation from Pandas to NumPy
    pandas_example_start = "df = pd.DataFrame({'X': [1, 2], 'Y': [3, 4]})"
    pandas_example_sol = "result = df['X'] * df['Y']"
    pandas_example_apis = [
        "pd.DataFrame",
        "column indexing",
        "element-wise multiplication",
    ]
    numpy_example_start = "arr = np.array([[1, 3], [2, 4]])"
    numpy_example_sol = "result = arr[:, 0] * arr[:, 1]"
    numpy_example_apis = [
        "np.array",
        "indexing with slicing",
        "element-wise multiplication",
    ]
    invalid = '{"API 1": "np.array", "API 2": "element-wise multiplication"}'
    return f"""You are a data science expert and code assistant tasked with identifying the possible APIs needed for translating a solution between {source_lib} library code to {target_lib} library code.
Your goal is to analyze the solution written in the {source_lib} library and identify the corresponding APIs in the target {target_lib} library solution code.

**Instructions**:
1. Read the code carefully and identify the APIs needed to perform the same operations in the target {target_lib} library.
2. Return the APIs strictly in a JSON array format. Do not include any additional information or comments.
3. If the APIs are not returned in a valid JSON array, the output will be considered incorrect.
4. Examples of valid and invalid outputs:
   - Valid: ["np.array", "element-wise multiplication", "slicing with indexing"]
   - Invalid: "np.array, element-wise multiplication, slicing with indexing" (missing JSON array brackets)
   - Invalid: {invalid} (not a JSON array)
5. The order of the operations in the array should reflect the order of importance or relevance to the solution.
6. If you are not sure about an API name, describe its functionality in simple terms and enclose it in double quotes.
7. The maximal number of APIs to be identified is 3, so only return the most relevant and important APIs.

Here is an example to help you understand the task:
- Source {source_lib} Start Code:
{pandas_example_start if source_lib == 'Pandas' else numpy_example_start}
- Source {source_lib} Solution Code:
{pandas_example_sol if source_lib == 'Pandas' else numpy_example_sol}

- Target {target_lib} Start Code:
{numpy_example_start if target_lib == 'NumPy' else pandas_example_start}
- Target {target_lib} APIs or Operations in Solution Code:
```json
{pandas_example_apis if source_lib == 'Pandas' else numpy_example_apis}
```

Now, please analyze the following code snippet:
- Source {source_lib} Start Code:
{source_start_code}
- Source {source_lib} Solution Code:
{source_sol_code}

- Target {target_lib} Example Input:
{target_input}
- Target {target_lib} Start Code:
{target_start_code}

- Target {target_lib} APIs or Operations in Solution Code:
Please Note only return the APIs in a JSON array format with the length of **3** at most.
```json
"""


def ds_rag_prompt(
    source_start_code,
    source_sol_code,
    source_lib,
    target_lib,
    target_input,
    target_start_code,
):
    return f"""You are a data assistant. Your task is to translate the given {source_lib} library code to {target_lib} library code.

- Source {source_lib} Start Code:
{source_start_code}
- Source {source_lib} Solution Code:
{source_sol_code}

Here are some {target_lib} APIs documentation that you might need to use, please refer to them when translating the query:
[APIs]

- Target {target_lib} Example Input:
{target_input}
- Target {target_lib} Start Code:
{target_start_code}
- Target {target_lib} Solution Code:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only fill in the code block with the {target_lib} solution code.
```Python
"""
