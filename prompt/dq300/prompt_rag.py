# Prompt Templates for Zero Shot Input & Output

PANDAS_FORMAT_NOTE = "You must save your final result into a variable named `result`."


# Helper function to determine query type
def _get_query_type(dialect):
    """Returns the query type based on the database dialect."""
    return "PostgreSQL query" if dialect == "PostgreSQL" else "Pandas Query"


def dq_rag_op_identify(
    source_query,
    source_dialect,
    target_dialect,
    source_schema,
    target_schema,
):
    """
    Generates a prompt for identifying operations inside a query

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
    postgresql_example = "SELECT name, COUNT(*) FROM employees WHERE department = 'Sales' GROUP BY name ORDER BY COUNT(*) DESC LIMIT 10;"
    pandas_example = """filtered_employees = employees[employees['department'] == 'Sales']
name_counts = filtered_employees.groupby('name').size().reset_index(name='count')
sorted_name_counts = name_counts.sort_values(by='count', ascending=False)
result = sorted_name_counts.head(10)"""
    return f"""You are a data assistant. Your task is to identify the operations present in the given {source_dialect} query. Return the operations from the following predefined list, and format your response as a JSON array.

**Available Operations**:
- Filtering Rows
- Filtering with Multiple Conditions
- Filtering with OR
- Joining Tables
- Left Join
- Right Join
- Aggregations
- Group By with Aggregation
- Complex Aggregation with Filtering
- Sorting
- Sorting with Aggregation
- Finding Unique Values
- Count Unique Values
- String Operations
- String Matching
- Date Conversion
- Date Extraction
- Date Range Filtering
- Calculating Percentages
- Handling NULL Values
- Handling NULL Values with Condition
- Conditional Aggregation
- Nested Queries
- Applying Functions
- Finding Maximum Value
- Finding Minimum Value
- Finding Averages
- Row Count
- Limit Rows
- Complex Joins and Aggregations


**Instructions**:
1. Read the query carefully and identify the operations it performs.
2. Match the operations to the options listed above. Only include operations from this list.
3. Return the operations in a JSON array format. Do not include any additional information or comments.
4. The order of the operations in the array should reflect the order they appear in the query.

Here is an example to help you understand the task:
- Source Query:
{postgresql_example if source_dialect == 'PostgreSQL' else pandas_example}
- Operations:
```json
["Filtering Rows", "Group By with Aggregation", "Sorting with Aggregation", "Limit Rows"]
```

Now, please analyze the following query:
- Source {source_dialect_query}:
{source_query}
- Operations:
```json
"""


def dq_rag_prompt(
    source_query, source_dialect, target_dialect, source_schema, target_schema
):
    """
    Generates a prompt for translating queries rag v1.

    Args:
        source_query (str): The source query in the source dialect.
        source_dialect (str): The source database dialect (e.g., PostgreSQL, Pandas).
        target_dialect (str): The target database dialect (e.g., PostgreSQL, Pandas).
        source_schema (str): The schema of the source database.
        target_schema (str): The schema of the target database.
        guidelines (str): Guidelines for the translation.
    Returns:
        str: A formatted prompt.
    """
    source_dialect_query = _get_query_type(source_dialect)
    target_dialect_query = _get_query_type(target_dialect)

    return f"""You are a data assistant. Your task is to translate the given query from {source_dialect} to {target_dialect}.

Here are some guidelines that show the differences in syntax and built-in functions between {source_dialect} and {target_dialect}. Please refer to them when translating the query:
[GUIDELINES]

Translate the above query into {target_dialect}. Ensure that:
• All data types are appropriately handled (e.g., convert date columns to datetime).
• Index alignment issues are addressed (e.g., reset indices when necessary).
• The schema is accurately reflected in the translation.
 
- **Source {source_dialect} Query**:
{source_query}
- **Target {target_dialect} Database Schema**:
{target_schema}

**Please Note:
1. Ensure that your translation adheres to the provided guidelines, utilizing the correct syntax and functions corresponding to the operations involved.
2. Do not include any import statements, comments, explanations, or additional information. 
3. Your response should contain only the translated code within the code block. {PANDAS_FORMAT_NOTE if target_dialect == 'Pandas' else ''}
4. Always use the columns provided for the target database.
4. Make sure the generated {target_dialect_query} is syntactically correct, executable and produces the expected results.
- **Target {target_dialect} Query**:
```{"SQL" if target_dialect == "PostgreSQL" else "Python"}
"""
