import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from contextlib import contextmanager
import os


# ---------------------------------------------------------
# DATABASE CONNECTIONS
# ---------------------------------------------------------

def get_sqlite_connection(db_path="database/inhibitors.db"):
    """
    Returns a SQLite connection.
    Creates directory if not existing.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def get_postgres_engine(user, password, host, port, db_name):
    """
    Returns SQLAlchemy engine for PostgreSQL.
    """
    connection_string = (
        f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    )
    return create_engine(connection_string)


# ---------------------------------------------------------
# CONTEXT MANAGER FOR SAFE CONNECTION HANDLING
# ---------------------------------------------------------

@contextmanager
def sqlite_cursor(db_path="database/inhibitors.db"):
    conn = get_sqlite_connection(db_path)
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# ---------------------------------------------------------
# TABLE CREATION
# ---------------------------------------------------------

def initialize_schema(db_path="database/inhibitors.db", schema_file="sql/schema.sql"):
    """
    Executes schema.sql to create tables.
    """
    with open(schema_file, "r") as f:
        schema_sql = f.read()

    with sqlite_cursor(db_path) as cursor:
        cursor.executescript(schema_sql)


# ---------------------------------------------------------
# BULK CSV LOAD INTO DATABASE
# ---------------------------------------------------------

def load_csv_to_table(csv_path, table_name, db_path="database/inhibitors.db"):
    """
    Loads CSV file into SQL table using pandas.
    """
    conn = get_sqlite_connection(db_path)
    df = pd.read_csv(csv_path)

    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()


# ---------------------------------------------------------
# QUERY EXECUTION
# ---------------------------------------------------------

def execute_query(query, db_path="database/inhibitors.db"):
    """
    Executes a SELECT query and returns pandas DataFrame.
    """
    conn = get_sqlite_connection(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def execute_sql_file(sql_file, db_path="database/inhibitors.db"):
    """
    Executes an entire SQL script file.
    """
    with open(sql_file, "r") as f:
        sql_script = f.read()

    with sqlite_cursor(db_path) as cursor:
        cursor.executescript(sql_script)


# ---------------------------------------------------------
# EXPORT QUERY RESULTS
# ---------------------------------------------------------

def export_query_to_csv(query, output_path, db_path="database/inhibitors.db"):
    """
    Executes query and saves result as CSV.
    """
    df = execute_query(query, db_path)
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------
# TABLE UTILITIES
# ---------------------------------------------------------

def get_table_names(db_path="database/inhibitors.db"):
    query = """
    SELECT name FROM sqlite_master
    WHERE type='table';
    """
    df = execute_query(query, db_path)
    return df["name"].tolist()


def count_rows(table_name, db_path="database/inhibitors.db"):
    query = f"SELECT COUNT(*) as row_count FROM {table_name};"
    df = execute_query(query, db_path)
    return df["row_count"].iloc[0]


def drop_table(table_name, db_path="database/inhibitors.db"):
    with sqlite_cursor(db_path) as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
