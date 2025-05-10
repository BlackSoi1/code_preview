def dl_rag_api_identify(
    source_start_code,
    source_sol_code,
    source_framework,
    target_framework,
    target_input,
    target_start_code,
):
    # Example for translation between PyTorch and TensorFlow
    pytorch_example_start = (
        "import torch\nx = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)"
    )
    pytorch_example_sol = "y = x * 2 + 1\ny.backward()"
    pytorch_example_apis = [
        "torch.tensor",
        "tensor operations (multiplication and addition)",
        "autograd.backward",
    ]

    tensorflow_example_start = (
        "import tensorflow as tf\nx = tf.Variable([1.0, 2.0, 3.0])"
    )
    tensorflow_example_sol = (
        "y = x * 2 + 1\ngradients = tf.GradientTape().gradient(y, x)"
    )
    tensorflow_example_apis = [
        "tf.Variable",
        "tensor operations (multiplication and addition)",
        "tf.GradientTape().gradient",
    ]

    invalid = '{"API 1": "tf.Variable", "API 2": "tf.GradientTape"}'
    return f"""You are a deep learning expert and code assistant tasked with identifying the possible APIs needed for translating a solution between {source_framework} framework code to {target_framework} framework code.
Your goal is to analyze the solution written in the {source_framework} framework and identify the corresponding APIs in the target {target_framework} framework solution code.

**Instructions**:
1. Read the code carefully and identify the APIs needed to perform the same operations in the target {target_framework} framework.
2. Return the APIs strictly in a JSON array format. Do not include any additional information or comments.
3. If the APIs are not returned in a valid JSON array, the output will be considered incorrect.
4. Examples of valid and invalid outputs:
   - Valid: 
   - Invalid: "np.array, element-wise multiplication, slicing with indexing" (missing JSON array brackets)
   - Invalid: {invalid} (not a JSON array)
5. The order of the operations in the array should reflect the order of importance or relevance to the solution.
6. If you are not sure about an API name, describe its functionality in simple terms and enclose it in double quotes.
7. The maximal number of APIs to be identified is 3, so only return the most relevant and important APIs.

Here is an example to help you understand the task:
- Source {source_framework} Start Code:
{pytorch_example_start if source_framework == 'PyTorch' else tensorflow_example_start}
- Source {source_framework} Solution Code:
{pytorch_example_sol if source_framework == 'PyTorch' else tensorflow_example_sol}

- Target {target_framework} Start Code:
{tensorflow_example_start if target_framework == 'TensorFlow' else pytorch_example_start}
- Target {target_framework} APIs in Solution Code:
```json
{pytorch_example_apis if source_framework == 'PyTorch' else tensorflow_example_apis}
```


Now, please analyze the following code snippet:
- Source {source_framework} Start Code:
{source_start_code}
- Source {source_framework} Solution Code:
{source_sol_code}

- Target {target_framework} Example Input:
{target_input}
- Target {target_framework} Start Code:
{target_start_code}

- Target {target_framework} APIs in Solution Code:
Please Note only return the APIs in a JSON array format with the length of **3** at most.
```json
"""


def dl_rag_prompt(
    source_start_code,
    source_sol_code,
    source_framework,
    target_framework,
    target_input,
    target_start_code,
):
    return f"""You are a deep learning engineer. Your task is to translate the given {source_framework} framework code to {target_framework} framework code.

- Source {source_framework} Start Code:
{source_start_code}
- Source {source_framework} Solution Code:
{source_sol_code}

Here are some {target_framework} APIs documentation that you might need to use, please refer to them when translating the query:
[APIs]

- Target {target_framework} Input Example:
{target_input}
- Target {target_framework} Start Code:
{target_start_code}
- Target {target_framework} Solution Code:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_framework} solution code.
```Python
"""
