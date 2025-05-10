def dl_zero_prompt_complete(
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

- Target {target_framework} Input Example:
{target_input}
- Target {target_framework} Start Code:
{target_start_code}
- Target {target_framework} Solution Code:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_framework} solution code.
```Python
"""
