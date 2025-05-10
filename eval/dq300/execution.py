import json
from contextlib import contextmanager
import sqlite3
from decimal import Decimal
import pandas as pd
import os
from typing import List, Dict, Any
import numpy as np
from psycopg2 import pool as pg_pool
import psycopg2
from datetime import date, datetime


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def format_float(old_res, digits=1):
    res = []
    for row in old_res:
        new_row = []
        for value in row:
            if (
                isinstance(value, float)
                or isinstance(value, Decimal)
                or isinstance(value, int)
            ):
                # Convert Decimal to float before rounding
                adjusted_value = (
                    float(value) + 1e-10
                )  # Add a small epsilon to mitigate precision issues
                new_row.append(round(adjusted_value, digits))
            elif isinstance(value, (date, datetime)):
                # Convert date or datetime to string in ISO format
                new_row.append(value.isoformat())
            else:
                new_row.append(value)
        res.append(tuple(new_row))
    return res


def execute_pandas_query(data_dir, table_list, query):
    """
    Executes a query using preloaded Pandas DataFrames saved as Parquet files.

    Args:
        data_dir (str): Directory containing pre-saved Parquet files.
        table_list (list): List of table names to load from Parquet files.
        query (str): Query string to execute. Should be valid Python code.

    Returns:
        list: The result of the executed query as a list of tuples.
    """
    # Load tables from Parquet files
    tables = {}
    for table_name in table_list:
        file_path = os.path.join(data_dir, f"{table_name}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Table {table_name} not found in {data_dir}")
        tables[table_name] = pd.read_parquet(file_path)

    # Create a local scope for exec
    local_scope = {"pd": pd, "np": np}
    local_scope.update(tables)  # Add the tables to the scope
    # Execute the query
    exec(query, {}, local_scope)

    # Retrieve the output variable 'result' from the local scope
    result = local_scope.get("result", None)

    # Convert result to a list of tuples
    if isinstance(result, pd.DataFrame):
        result = [tuple(row) for row in result.to_records(index=False)]
    elif isinstance(result, pd.Series):
        result = [(value,) for value in result.tolist()]
    elif isinstance(result, np.ndarray):
        result = [
            tuple(row) if isinstance(row, (np.ndarray, list)) else (row,)
            for row in result
        ]
    elif isinstance(result, (list, tuple)):
        result = (
            [tuple(result)]
            if isinstance(result, tuple)
            else [
                tuple(item) if isinstance(item, (list, np.ndarray)) else (item,)
                for item in result
            ]
        )
    elif isinstance(result, (int, float, str, np.bool_, np.int64, np.float64)):
        result = [(result,)]
    else:
        raise TypeError(f"Unsupported result type: {type(result)}")

    return result


# PostgreSQL connection pool
pg_pool = pg_pool.SimpleConnectionPool(
    1,
    20,
    dbname="DB_NAME",
    user="root",
    host="localhost",
    password="PASSWORD",
    port="5432",
)


def connect_to_database():
    """Connect to the PostgreSQL database"""
    try:
        conn = pg_pool.getconn()
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def execute_sql_query(conn, query):
    """Execute an SQL query on the given database connection."""
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except psycopg2.Error as e:
        # print(f"Error executing SQL query: {e}")
        return None


def close_connection(conn):
    """Close the connection to the database."""
    try:
        pg_pool.putconn(conn)
        # print("Connection closed successfully.")
    except psycopg2.Error as e:
        print(f"Error closing connection: {e}")


def perform_query_on_postgre_databases(query):
    connection = connect_to_database()
    res = execute_sql_query(connection, query)
    close_connection(connection)
    return res
