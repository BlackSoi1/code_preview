# Prompt Templates for Zero Shot Input & Output

PANDAS_FORMAT_NOTE = "You must save your final result into a variable named `result`."


# Helper function to determine query type
def _get_query_type(dialect):
    """Returns the query type based on the database dialect."""
    return "PostgreSQL query" if dialect == "PostgreSQL" else "Pandas Query"


# Intent Extraction Prompt Template
def dq_extract_intent(
    source_query,
    source_dialect,
    target_dialect,
    source_schema,
    target_schema,
):
    """
    Generates a prompt for extracting the intent behind the given source query.

    Args:
        source_query (str): The source query in the source dialect.
        source_dialect (str): The source database dialect (e.g., PostgreSQL, Pandas).
        source_schema (str): The schema of the source database.

    Returns:
        str: A formatted prompt for intent extraction.
    """
    source_dialect_query = (
        "PostgreSQL query" if source_dialect == "PostgreSQL" else "Pandas Query"
    )

    return f"""You are a data assistant. Your task is to analyze the given {source_dialect_query} and extract the intent behind it. The intent should describe the high-level operation or purpose of the query.

**Instructions**:
1. Carefully read the {source_dialect_query} and understand its structure and purpose.
2. Identify the high-level intent behind the query. For example:
   - "Filter rows where column X is greater than 100."
   - "Aggregate data by column Y and calculate the sum of column Z."
   - "Join two tables on column A and column B."
   - "Sort rows by column X in descending order."
   - "Group data by column Y and apply the COUNT function."
   - "Create a new column by performing arithmetic on existing columns."
3. Output the identified intent as a single concise sentence. Do not include code, comments, or explanations.

- Source {source_dialect_query}:
{source_query}

- Source {source_dialect} database schema:
{source_schema}

- Identified Intent:
[YOUR INTENT HERE]
"""


cot_intent_extract = """You are a data assisant. Your task is to analyze the given [[source_library]] Query and extract the intent behind it.

- Source [[source_library]] Query:
[[source_code]]

- Source [[source_library]] database schema:
[[source_schema]]

Let's think step by step:
-- Step1: """


def dq_intent_prompt(
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
        str: A formatted prompt.INTENT
    """
    source_dialect_query = _get_query_type(source_dialect)
    target_dialect_query = _get_query_type(target_dialect)

    return f"""You are a data assistant. Your task is to translate the given {source_dialect_query} to {target_dialect_query}.

- Intent:
[INTENT]

- Source {source_dialect_query}:
{source_query}
- Target {target_dialect} database schema:
{target_schema}
- Target {target_dialect_query}:
Please Note: do not include any import statements, comments, explanations, or additional information. Your response should only contain the {target_dialect_query}. {PANDAS_FORMAT_NOTE if target_dialect == 'Pandas' else ''}
```{"SQL" if target_dialect == "PostgreSQL" else "Python"}
"""
