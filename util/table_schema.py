import sqlite3
import pandas as pd
import psycopg2

# A mapping from database identifiers to their respective tables
db_table_map = {
    "debit_card_specializing": [
        "customers",
        "gasstations",
        "products",
        "transactions_1k",
        "yearmonth",
    ],
    "student_club": [
        "major",
        "member",
        "attendance",
        "budget",
        "event",
        "expense",
        "income",
        "zip_code",
    ],
    "thrombosis_prediction": ["Patient", "Examination", "Laboratory"],
    "european_football_2": [
        "League",
        "Match",
        "Player",
        "Player_Attributes",
        "Team",
        "Team_Attributes",
        "Country",
    ],
    "formula_1": [
        "circuits",
        "seasons",
        "races",
        "constructors",
        "constructorResults",
        "constructorStandings",
        "drivers",
        "driverStandings",
        "lapTimes",
        "pitStops",
        "qualifying",
        "status",
        "results",
    ],
    "superhero": [
        "alignment",
        "attribute",
        "colour",
        "gender",
        "publisher",
        "race",
        "superpower",
        "superhero",
        "hero_attribute",
        "hero_power",
    ],
    "codebase_community": [
        "posts",
        "users",
        "badges",
        "comments",
        "postHistory",
        "postLinks",
        "tags",
        "votes",
    ],
    "card_games": [
        "cards",
        "foreign_data",
        "legalities",
        "rulings",
        "set_translations",
        "sets",
    ],
    "toxicology": ["molecule", "atom", "bond", "connected"],
    "california_schools": ["satscores", "frpm", "schools"],
    "financial": [
        "district",
        "account",
        "client",
        "disp",
        "card",
        "loan",
        "order",
        "trans",
    ],
}


def find_table_in_sql(sql, db_id):
    """
    Identify which tables from a given database ID appear in a given SQL query.

    Args:
        sql (str): The SQL query string.
        db_id (str): The identifier of the database (must be a key in db_table_map).

    Returns:
        list: A list of table names that appear in the SQL query.
    """
    tables = db_table_map.get(db_id, [])
    return [table for table in tables if table in sql]


def nice_look_table(column_names, values):
    """
    Format a table (column names and values) into a nicely aligned string.

    Args:
        column_names (list): List of column names.
        values (list): List of rows, each a list of values.

    Returns:
        str: A formatted string representing a table with aligned columns.
    """
    # Determine the maximum width of each column
    widths = [
        max(len(str(value[i])) for value in values + [column_names])
        for i in range(len(column_names))
    ]

    # Format the header
    header = "".join(
        f"{column.rjust(width)} " for column, width in zip(column_names, widths)
    )

    # Format the rows
    rows = []
    for value in values:
        row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
        rows.append(row)

    return header + "\n" + "\n".join(rows)


def generate_schema_prompt_pandas(db_path):
    """
    Describe the schema of all tables in a SQLite database using Pandas.
    Includes the column names and data types for each table.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        str: A formatted string describing all table schemas with columns and their data types.
    """
    connection = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, connection)["name"].tolist()

    schema_descriptions = []
    for table_name in tables:
        # Skip SQLite's internal sequence table
        if table_name == "sqlite_sequence":
            continue

        query = f"SELECT * FROM `{table_name}` LIMIT 3;"
        df = pd.read_sql_query(query, connection)

        schema_description = f"Table: {table_name}\nColumns:\n"
        for column_name, dtype in zip(df.columns, df.dtypes):
            schema_description += f"  - {column_name}: {dtype}\n"

        schema_descriptions.append(schema_description)

    connection.close()
    return "\n\n".join(schema_descriptions)


def connect_postgresql():
    """
    Connect to a local PostgreSQL database named 'DB_NAME' using hardcoded credentials.

    Returns:
        psycopg2.extensions.connection: A connection object to the PostgreSQL database.
    """
    db = psycopg2.connect(
        "dbname=DB_NAME user=root host=localhost password=PASSWORD port=5432"
    )
    return db


def format_postgresql_create_table(
    table_name, columns_info, primary_keys, foreign_keys
):
    """
    Format a PostgreSQL CREATE TABLE statement based on the given schema information.

    Args:
        table_name (str): Name of the table.
        columns_info (list): A list of tuples (column_name, data_type, is_nullable, column_default).
        primary_keys (list or None): List of primary key columns.
        foreign_keys (list or None): List of foreign key constraints as tuples (fk_column, ref_table, ref_column).

    Returns:
        str: A CREATE TABLE statement.
    """
    lines = [f"CREATE TABLE {table_name} ("]
    for i, column in enumerate(columns_info):
        column_name, data_type, is_nullable, column_default = column
        null_status = "NULL" if is_nullable == "YES" else "NOT NULL"
        default = f"DEFAULT {column_default}" if column_default else ""
        column_line = f"    {column_name} {data_type} {null_status} {default}".strip()

        # Add a comma if there are more columns, primary keys, or foreign keys following
        if i < len(columns_info) - 1 or primary_keys or foreign_keys:
            column_line += ","
        lines.append(column_line)

    if primary_keys:
        pk_line = f"    PRIMARY KEY ({', '.join(primary_keys)})"
        if foreign_keys:
            pk_line += ","
        lines.append(pk_line)

    if foreign_keys:
        for fk in foreign_keys:
            fk_column, ref_table, ref_column = fk
            fk_line = (
                f"    FOREIGN KEY ({fk_column}) REFERENCES {ref_table}({ref_column})"
            )
            if fk != foreign_keys[-1]:
                fk_line += ","
            lines.append(fk_line)

    lines.append(");")
    return "\n".join(lines)


def generate_schema_prompt_postgresql(db_path, table_name=None):
    """
    Generate a CREATE TABLE schema prompt for all tables in a PostgreSQL database
    based on a given SQLite database path (used to identify which database to query).

    Args:
        db_path (str): Path to a SQLite database file, used to extract the database name.
        table_name (str, optional): If provided, only generate the schema for this specific table.

    Returns:
        str: A combined schema prompt containing CREATE TABLE statements for the requested tables.
    """
    db_name = db_path.split("/")[-1].split(".sqlite")[0]
    db = connect_postgresql()
    cursor = db.cursor()
    tables = db_table_map.get(db_name, [])
    schemas = {}

    for table in tables:
        if table_name and table_name != table:
            continue

        # Get column information
        cursor.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
            """,
            (table,),
        )
        columns_info = cursor.fetchall()

        # Get primary key information
        cursor.execute(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid
            AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary;
            """,
            (table,),
        )
        primary_keys = [row[0] for row in cursor.fetchall()]

        # Get foreign key information
        cursor.execute(
            """
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
            """,
            (table,),
        )
        foreign_keys = cursor.fetchall()

        pretty_schema = format_postgresql_create_table(
            table, columns_info, primary_keys, foreign_keys
        )
        schemas[table] = pretty_schema

    schema_prompt = "\n\n".join(schemas.values())
    cursor.close()
    db.close()
    return schema_prompt
