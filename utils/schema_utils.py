"""
Pandora Schema Utilities

Utilities for formatting and processing database/KG schema information,
including primary key and foreign key constraint formatting.
"""

import os
from typing import Tuple, Dict, Optional

import pandas as pd

from utils.file_utils import load_json


def wrap_primary_foreign_keys_for_db(db_data: dict) -> Tuple[str, str]:
    """
    Extract and format primary and foreign key constraints from database metadata.

    Processes structured database schema information to produce human-readable constraints.
    Handles both single-column and composite (multi-column) keys.

    Args:
        db_data (dict): Database metadata dictionary with keys:
            - db_id (str): Database identifier.
            - table_names_original (list[str]): List of table names.
            - column_names_original (list[tuple[int, str]]): List of (table_id, column_name) tuples.
            - primary_keys (list): Primary key column indices. Each element is either a single int (single-column key) or a list of ints (composite key).
            - foreign_keys (list[tuple[int, int]]): Foreign key relationships as pairs of (from_column_id, to_column_id).

    Returns:
        tuple[str, str]: A tuple of (wrapped_primary_keys, wrapped_foreign_keys).
            - wrapped_primary_keys (str): Newline-separated primary key constraints.
              Format: TABLE `{table_name}` PRIMARY KEY `{column_names}`
            - wrapped_foreign_keys (str): Newline-separated foreign key constraints.
              Format: FOREIGN KEY {from_table}['{from_column}'] REFERENCES {to_table}['{to_column}']

    Example:
        >>> db_data = {
        ...     'db_id': 'company',
        ...     'table_names_original': ['employees', 'departments'],
        ...     'column_names_original': [(0, 'emp_id'), (0, 'dept_id'), (1, 'dept_id')],
        ...     'primary_keys': [0],
        ...     'foreign_keys': [(1, 2)]
        ... }
        >>> pks, fks = wrap_primary_foreign_keys_for_db(db_data)
        >>> print(pks)
        TABLE `employees` PRIMARY KEY `emp_id`
        >>> print(fks)
        FOREIGN KEY employees['dept_id'] REFERENCES departments['dept_id']
    """
    table_names = db_data["table_names_original"]
    column_names = db_data["column_names_original"]

    # ───────────────────────────────────────────────────────────────────────
    # Extract and format primary keys
    # ───────────────────────────────────────────────────────────────────────
    primary_keys: dict[str, str] = {}
    for pk_id in db_data["primary_keys"]:
        if isinstance(pk_id, int):
            # Single-column primary key
            table_idx, column_name = column_names[pk_id]
            table_name = table_names[table_idx]
            primary_keys[table_name] = column_name
        elif isinstance(pk_id, list):
            # Composite (multi-column) primary key
            columns = []
            table_name = ""
            for col_id in pk_id:
                table_idx, column_name = column_names[col_id]
                table_name = table_names[table_idx]
                columns.append(column_name)
            primary_keys[table_name] = ", ".join(columns)

    # Format primary keys as readable strings
    wrapped_primary_keys = [
        f"TABLE `{table_name}` PRIMARY KEY `{column_list}`"
        for table_name, column_list in primary_keys.items()
    ]
    wrapped_primary_keys_str = "\n".join(wrapped_primary_keys)

    # ───────────────────────────────────────────────────────────────────────
    # Extract and format foreign keys
    # ───────────────────────────────────────────────────────────────────────
    wrapped_foreign_keys = []
    for from_col_id, to_col_id in db_data["foreign_keys"]:
        # Resolve source column
        from_table_idx, from_column_name = column_names[from_col_id]
        from_table_name = table_names[from_table_idx]

        # Resolve target (referenced) column
        to_table_idx, to_column_name = column_names[to_col_id]
        to_table_name = table_names[to_table_idx]

        wrapped_foreign_keys.append(
            f"FOREIGN KEY {from_table_name}['{from_column_name}'] REFERENCES {to_table_name}['{to_column_name}']"
        )

    wrapped_foreign_keys_str = "\n".join(wrapped_foreign_keys)

    return wrapped_primary_keys_str, wrapped_foreign_keys_str


def wrap_primary_foreign_keys_for_kg(box_path: str) -> Tuple[str, str]:
    """
    Extract and format primary and foreign key constraints from Knowledge Graph CSV files.

    Processes CSV files in a directory to infer primary keys (first column of each CSV)
    and loads foreign key relationships from a foreign_key.json file.

    Args:
        box_path (str): Path to directory containing KG CSV files and foreign_key.json

    Returns:
        tuple[str, str]: A tuple of (wrapped_primary_keys, wrapped_foreign_keys).
            - wrapped_primary_keys (str): Newline-separated primary key constraints.
              Format: TABLE `{table_name}` PRIMARY KEY `{column_name}`
            - wrapped_foreign_keys (str): Newline-separated foreign key constraints.
              Format: FOREIGN KEY {from_table}['{from_column}'] REFERENCES {to_table}['{to_column}']
    """
    def get_first_column_name(csv_path: str) -> str:
        """Get the first column name from a CSV file."""
        df = pd.read_csv(csv_path, nrows=0)
        first_col = df.columns[0]
        return first_col

    primary_keys: dict[str, str] = {}
    for filename in os.listdir(box_path):
        if filename.lower().endswith(".csv"):
            table_name = os.path.splitext(filename)[0].split(".")[-1]
            first_column_name = get_first_column_name(os.path.join(box_path, filename))
            primary_keys[table_name] = first_column_name

    # Format primary keys as readable strings
    wrapped_primary_keys = [
        f"TABLE `{table_name}` PRIMARY KEY `{column_list}`"
        for table_name, column_list in primary_keys.items()
    ]
    wrapped_primary_keys_str = "\n".join(wrapped_primary_keys)

    # ───────────────────────────────────────────────────────────────────────
    # Extract and format foreign keys
    # ───────────────────────────────────────────────────────────────────────
    wrapped_foreign_keys = []
    foreign_keys_file = os.path.join(box_path, "foreign_key.json")
    if os.path.exists(foreign_keys_file):
        foreign_keys = load_json(foreign_keys_file)
        for from_col, to_col in foreign_keys:
            from_table_name, from_column_name = from_col.split("-")
            from_table_name = from_table_name.split(".")[-1]
            to_table_name, to_column_name = to_col.split("-")
            to_table_name = to_table_name.split(".")[-1]
            wrapped_foreign_keys.append(
                f"FOREIGN KEY {from_table_name}['{from_column_name}'] REFERENCES {to_table_name}['{to_column_name}']"
            )

    wrapped_foreign_keys_str = "\n".join(wrapped_foreign_keys)

    return wrapped_primary_keys_str, wrapped_foreign_keys_str


def load_column_descriptions(db_dir: str) -> dict[str, str]:
    """
    Load column descriptions from BIRD database description CSV files.

    Processes description files for each database and creates a mapping of
    column IDs to their descriptions.

    Args:
        db_dir: Path to database directory containing db_id subdirectories
                with 'database_description' folders inside.

    Returns:
        Dictionary mapping column_id to description string.
        Column ID format: "{db_id}.{table_name}.{column_name}"

    Example:
        {
            "california_schools.frpms.Free_Rate": "(float), Free_Rate, Percentage of free meals",
            "california_schools.frpms.Count": "(int), Count, Number of students"
        }
    """
    import tqdm

    column_description = {}

    for db_id in tqdm.tqdm(os.listdir(db_dir)):
        db_path = os.path.join(db_dir, db_id)
        if not os.path.isdir(db_path):
            continue

        desc_dir = os.path.join(db_path, 'database_description')
        if not os.path.exists(desc_dir):
            continue

        for file in os.listdir(desc_dir):
            if file.endswith(".csv"):
                # Handle file name corrections
                if file == "set_transactions.csv":
                    file = "set_translations.csv"
                if file == "ruling.csv":
                    file = "rulings.csv"

                csv_path = os.path.join(desc_dir, file)
                try:
                    df = pd.read_csv(csv_path, encoding='latin1')
                    # Clean column names
                    df.columns = df.columns.str.replace('ï»¿', '').str.strip()

                    table_name = file.replace(".csv", "")

                    for i in range(len(df)):
                        if pd.isna(df['original_column_name'][i]):
                            continue

                        column_name = df['original_column_name'][i].strip()
                        column_id = f"{db_id}.{table_name}.{column_name}"

                        # Handle special cases
                        if db_id == "student_club":
                            column_id = column_id.lower()

                        wrapped_column_name = df['column_name'][i] if not pd.isna(df['column_name'][i]) else column_name.lower()

                        # Build description
                        data_format = df['data_format'][i] if not pd.isna(df['data_format'][i]) else "unknown"

                        if pd.isna(df['column_description'][i]):
                            column_description[column_id] = f"({data_format}), {wrapped_column_name}"
                        else:
                            column_description[column_id] = f"({data_format}), {wrapped_column_name}, {df['column_description'][i]}"

                except Exception as e:
                    print(f"Warning: Failed to load {csv_path}: {e}")

    return column_description


def build_box_schema_with_descriptions(tables_df: Dict[str, pd.DataFrame], column_descriptions: dict, db_id: str) -> str:
    """
    Build box_schema with column descriptions as Python comments.

    Creates DataFrame schema definitions with each column followed by
    a comment containing its description.

    Args:
        tables_df: Dictionary mapping table names to DataFrames
        column_descriptions: Column description dictionary from load_column_descriptions()
        db_id: Database ID for looking up descriptions

    Returns:
        Formatted schema string with comments.

    Example Output:
        frpms = pd.DataFrame({
            "Free_Rate": [],  # (float), Free_Rate, Percentage of free meals
            "Count": [],  # (int), Count, Number of students
        })
    """
    lines = []
    for table_name, df in tables_df.items():
        lines.append(f"{table_name} = pd.DataFrame({{")

        for col in df.columns:
            # Look up column description
            col_key = f"{db_id}.{table_name}.{col}"
            description = column_descriptions.get(col_key, "")

            if description:
                lines.append(f'    "{col}": [],  # {description}')
            else:
                lines.append(f'    "{col}": [],')

        lines.append("})")
        lines.append("")

    return "\n".join(lines)


def load_kg_dataframe(db_id: str, db_dir: str, top_k_row: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Load Knowledge Graph data from CSV files.

    Args:
        db_id: Database/KG identifier
        db_dir: Directory containing KG subdirectories
        top_k_row: Number of rows to load from each CSV

    Returns:
        Dictionary mapping table names to DataFrames
    """
    tables = {}
    kg_path = os.path.join(db_dir, db_id)

    if not os.path.exists(kg_path):
        return tables

    for filename in os.listdir(kg_path):
        if filename.lower().endswith(".csv"):
            table_name = os.path.splitext(filename)[0]
            csv_path = os.path.join(kg_path, filename)
            try:
                df = pd.read_csv(csv_path, nrows=top_k_row)
                tables[table_name] = df
            except Exception as e:
                print(f"Warning: Failed to load {csv_path}: {e}")

    return tables


def load_table_dataframe(table_path: str, top_k_row: int = 3) -> pd.DataFrame:
    """
    Load a single table from CSV or other supported formats.

    Args:
        table_path: Path to table file
        top_k_row: Number of rows to load

    Returns:
        DataFrame containing table data
    """
    try:
        if table_path.endswith('.csv'):
            return pd.read_csv(table_path, nrows=top_k_row)
        elif table_path.endswith('.xlsx'):
            return pd.read_excel(table_path, nrows=top_k_row)
        else:
            print(f"Warning: Unsupported table format: {table_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Failed to load {table_path}: {e}")
        return pd.DataFrame()


def prepare_kb_info(
    kb_type: str,
    kb_id: str,
    data_dir: str,
    schema_data: Optional[dict] = None,
    column_descriptions: Optional[dict] = None,
    top_k_row: int = 3
) -> dict:
    """
    Unified function to prepare knowledge base information for different data sources.

    Supports three types of knowledge sources:
    - 'db': SQLite database with schema metadata
    - 'kg': Knowledge Graph with CSV files
    - 'table': Single table file

    Args:
        kb_type: Type of knowledge base ('db', 'kg', 'table')
        kb_id: Identifier for the knowledge base
        data_dir: Root directory containing the data
        schema_data: Optional schema metadata (for DB type)
        column_descriptions: Optional column descriptions dictionary (for DB type)
        top_k_row: Number of sample rows to load

    Returns:
        Dictionary containing:
        - box_schema: Pandas DataFrame schema definition
        - table_df: Dict of table name -> DataFrame (or single DataFrame for 'table')
        - table_content: Formatted string of table previews
        - primary_keys: Formatted primary key constraints
        - foreign_keys: Formatted foreign key constraints
        - kb_path: Path to the knowledge base
    """
    if kb_type == 'db':
        return _prepare_db_kb_info(kb_id, data_dir, schema_data, column_descriptions, top_k_row)
    elif kb_type == 'kg':
        return _prepare_kg_kb_info(kb_id, data_dir, top_k_row)
    elif kb_type == 'table':
        return _prepare_table_kb_info(kb_id, data_dir, top_k_row)
    else:
        raise ValueError(f"Unknown kb_type: {kb_type}. Supported: 'db', 'kg', 'table'")


def _prepare_db_kb_info(db_id: str, db_dir: str, schema_data: Optional[dict], column_descriptions: Optional[dict] = None, top_k_row: int = 3) -> dict:
    """
    Prepare knowledge base info for SQLite database.

    Args:
        db_id: Database identifier
        db_dir: Directory containing database files
        schema_data: Schema metadata dictionary
        column_descriptions: Optional column descriptions dictionary
        top_k_row: Number of sample rows to load

    Returns:
        Knowledge base info dictionary
    """
    import sqlite3

    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Load table DataFrames
    tables_df = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]

        for table_name in table_names:
            query = f'SELECT * FROM "{table_name}" LIMIT {top_k_row}'
            tables_df[table_name] = pd.read_sql_query(query, conn)

        conn.close()
    except Exception as e:
        raise Exception(f"Failed to load tables from {db_path}: {e}")

    # Build box schema with descriptions if available
    box_schema = build_box_schema_with_descriptions(tables_df, column_descriptions or {}, db_id)

    # Build table content
    content_lines = []
    for table_name, df in tables_df.items():
        content_lines.append(f"TABLE `{table_name}`:\n{str(df)}")
    table_content = "\n\n".join(content_lines)

    # Extract primary and foreign keys
    primary_keys = ""
    foreign_keys = ""
    if schema_data:
        primary_keys, foreign_keys = wrap_primary_foreign_keys_for_db(schema_data)

    return {
        "box_schema": box_schema,
        "table_df": tables_df,
        "table_content": table_content,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "db_path": db_path,
        "kb_type": "db",
    }


def _prepare_kg_kb_info(kg_id: str, db_dir: str, top_k_row: int) -> dict:
    """
    Prepare knowledge base info for Knowledge Graph (CSV files).

    Args:
        kg_id: Knowledge Graph identifier
        db_dir: Directory containing KG subdirectories
        top_k_row: Number of sample rows to load

    Returns:
        Knowledge base info dictionary
    """
    kg_path = os.path.join(db_dir, kg_id)

    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"KG directory not found: {kg_path}")

    # Load table DataFrames
    tables_df = load_kg_dataframe(kg_id, db_dir, top_k_row)

    # Build box schema
    lines = []
    for table_name, df in tables_df.items():
        lines.append(f"{table_name} = pd.DataFrame({{")
        col_lines = [f'    "{col}": [],' for col in df.columns]
        lines.append("\n".join(col_lines))
        lines.append("})")
        lines.append("")
    box_schema = "\n".join(lines)

    # Build table content
    content_lines = []
    for table_name, df in tables_df.items():
        content_lines.append(f"TABLE `{table_name}`:\n{str(df)}")
    table_content = "\n\n".join(content_lines)

    # Extract primary and foreign keys
    primary_keys, foreign_keys = wrap_primary_foreign_keys_for_kg(kg_path)

    return {
        "box_schema": box_schema,
        "table_df": tables_df,
        "table_content": table_content,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "db_path": kg_path,
        "kb_type": "kg",
    }


def _prepare_table_kb_info(table_id: str, table_path: str, top_k_row: int) -> dict:
    """
    Prepare knowledge base info for a single table file.

    Args:
        table_id: Table identifier
        table_path: Path to table file (CSV, Excel, etc.)
        top_k_row: Number of sample rows to load

    Returns:
        Knowledge base info dictionary
    """
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Table file not found: {table_path}")

    # Load table DataFrame
    df = load_table_dataframe(table_path, top_k_row)
    table_name = os.path.splitext(os.path.basename(table_path))[0]
    tables_df = {table_name: df}

    # Build box schema
    lines = []
    lines.append(f"{table_name} = pd.DataFrame({{")
    col_lines = [f'    "{col}": [],' for col in df.columns]
    lines.append("\n".join(col_lines))
    lines.append("})")
    lines.append("")
    box_schema = "\n".join(lines)

    # Build table content
    table_content = f"TABLE `{table_name}`:\n{str(df)}"

    return {
        "box_schema": box_schema,
        "table_df": tables_df,
        "table_content": table_content,
        "primary_keys": "",
        "foreign_keys": "",
        "db_path": table_path,
        "kb_type": "table",
    }
