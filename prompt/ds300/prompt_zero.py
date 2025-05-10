def ds_zero_prompt_complete(
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

- Target {target_lib} Example Input:
{target_input}
- Target {target_lib} Start Code:
{target_start_code}
- Target {target_lib} Solution Code:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_lib} solution code.
```Python
"""
