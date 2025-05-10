# Prompt Templates for Zero Shot Input & Output

PANDAS_FORMAT_NOTE = "You must save your final result into a variable named `result`."


# Helper function to determine query type
def _get_query_type(dialect):
    """Returns the query type based on the database dialect."""
    return "PostgreSQL query" if dialect == "PostgreSQL" else "Pandas Query"


def dq_zero_prompt_complete(
    source_query,
    source_dialect,
    target_dialect,
    source_schema,
    target_schema,
):
    """
    Generates a prompt for translating queries (Baseline V21).

    Args:
        source_query (str): The source query in the source dialect.
        source_dialect (str): The source database dialect (e.g., PostgreSQL, Pandas).
        target_dialect (str): The target database dialect (e.g., PostgreSQL, Pandas).
        source_schema (str): The schema of the source database.
        target_schema (str): The schema of the target database.

    Returns:
        str: A formatted prompt.
    """
    source_dialect_query = _get_query_type(source_dialect)
    target_dialect_query = _get_query_type(target_dialect)

    return f"""You are a data assistant. Your task is to translate the given {source_dialect_query} to {target_dialect_query}.

- Source {source_dialect_query}:
{source_query}
- Target {target_dialect} database schema:
{target_schema}
- Target {target_dialect_query}:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_dialect_query}. {PANDAS_FORMAT_NOTE if target_dialect == 'Pandas' else ''}
```{"SQL" if target_dialect == "PostgreSQL" else "Python"}
"""
