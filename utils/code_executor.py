"""
Pandora Code Executor

Unified code execution module for running generated Pandas/Python code
against SQLite databases with sandboxed execution.
"""

import ast
import sqlite3
import pandas as pd
import pickle
import subprocess
import tempfile
import os
import time
import traceback
import sys
from pathlib import Path
from typing import Any, Optional, Dict, Tuple

from utils.logger import setup_logger

logger = setup_logger("pandora.code_executor")


class SecurityViolation(ValueError):
    """Raised when generated code attempts an operation outside the sandbox policy."""


class GeneratedCodePolicy(ast.NodeVisitor):
    """Reject filesystem, network, process, reflection, and dynamic-code access."""

    ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "datetime", "statistics"}
    BLOCKED_NAMES = {
        "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
        "globals", "locals", "vars", "getattr", "setattr", "delattr", "help",
    }
    BLOCKED_ATTRIBUTES = {
        "system", "popen", "spawn", "fork", "forkpty", "kill", "remove",
        "unlink", "rmdir", "mkdir", "makedirs", "rename", "replace", "chmod",
        "chown", "walk", "listdir", "scandir", "read_csv", "read_excel",
        "read_json", "read_pickle", "read_sql", "read_parquet", "read_html",
        "to_csv", "to_excel", "to_json", "to_pickle", "to_parquet", "to_sql",
        "dump", "dumps", "load", "loads", "connect", "request", "urlopen",
    }

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in self.ALLOWED_IMPORTS:
                raise SecurityViolation(f"Import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root not in self.ALLOWED_IMPORTS:
            raise SecurityViolation(f"Import is not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.BLOCKED_NAMES or node.id.startswith("__"):
            raise SecurityViolation(f"Name is not allowed: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr in self.BLOCKED_ATTRIBUTES:
            raise SecurityViolation(f"Attribute is not allowed: {node.attr}")
        self.generic_visit(node)


def validate_generated_code(code: str) -> None:
    """Validate generated Python before it reaches the interpreter."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax failures are execution feedback, not policy violations.
        return
    GeneratedCodePolicy().visit(tree)


class ExecutionResult:
    """
    Structured execution result.

    Attributes:
        success: Whether execution succeeded
        result: Execution result (list of lists)
        error: Error message if failed
        execution_time: Execution time in seconds
        stdout: Standard output from execution
        df_vars: String describing newly defined variables (DataFrame, Series, and scalars)
        is_empty: Whether result is empty (success=True but no data returned)
    """

    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        stdout: str = "",
        df_vars: str = "",
        is_empty: bool = False,
    ):
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.stdout = stdout
        self.df_vars = df_vars
        self.is_empty = is_empty

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "stdout": self.stdout,
            "df_vars": self.df_vars,
            "is_empty": self.is_empty,
        }


# Default traceback template for code execution
DEFAULT_PROMPT_FOR_TRACEBACK_CODE = """
import traceback
import pickle
import sys
import pandas as pd

df_file_path = sys.argv[1]
result_file_path = sys.argv[2]

# Load DataFrames from pickle file
with open(df_file_path, 'rb') as file:
    dfs_dict = pickle.load(file)
    # Make each table available as a global variable
    for key, value in dfs_dict.items():
        globals()[key] = value

# Snapshot of pre-existing global keys (loaded tables)
initial_keys = set(globals().keys())

# Initialize result so it's always defined
result = None
df_vars_str = ''

try:
{{main_code}}

except Exception as e:
    temp_result = traceback.format_exc()
    result = ['Execution failed on python: \\n' + temp_result.split('\\n')[-2], temp_result]

# Capture newly defined variables (DataFrames, Series, and scalars)
try:
    # Exclude executor-internal variables from the "newly defined" set.
    # These are defined in the wrapper AFTER the initial_keys snapshot,
    # so they would otherwise leak into the Execution State output.
    _executor_internal = {'result', 'df_vars_str', 'initial_keys', 'temp_result', '_executor_internal'}
    new_keys = set(globals().keys()) - initial_keys - _executor_internal
    df_vars = []
    _nl = "\\n"
    for name in sorted(new_keys):
        obj = globals()[name]
        if isinstance(obj, pd.DataFrame):
            head = obj.head(3)
            col_info = []
            for col in head.columns:
                dtype = str(head[col].dtype)
                sample_vals = head[col].dropna().head(3).tolist()
                col_info.append(f"    {col} ({dtype}): {sample_vals}")
            head_str = str(head)
            df_vars.append(
                f"DataFrame '{name}' — shape {obj.shape}" + _nl
                + "  columns:" + _nl
                + _nl.join(col_info)
                + _nl + "  head(3):" + _nl + head_str
            )
        elif isinstance(obj, pd.Series):
            head_vals = obj.head(3).tolist()
            df_vars.append(
                f"Series '{name}' — dtype {obj.dtype}, length {len(obj)}" + _nl
                + f"  head(3): {head_vals}"
            )
        else:
            # Scalar: int, float, str, bool, list, dict, etc.
            try:
                # Special handling: zip objects from .itertuples() are not useful
                # Convert them to list so the LLM can see the actual content
                if type(obj).__name__ == 'zip':
                    try:
                        obj_list = list(obj)
                        val_repr = repr(obj_list)
                        if len(val_repr) > 100:
                            val_repr = val_repr[:97] + "..."
                        df_vars.append(f"Scalar '{name}' = {val_repr}")
                    except Exception:
                        df_vars.append(f"Scalar '{name}' = <zip object>")
                    continue
                val_repr = repr(obj)
                # Truncate long representations
                if len(val_repr) > 100:
                    val_repr = val_repr[:97] + "..."
                df_vars.append(f"Scalar '{name}' = {val_repr}")
            except Exception:
                df_vars.append(f"Scalar '{name}' = <unrepr>")
    if df_vars:
        _sep = _nl + "---" + _nl
        df_vars_str = _sep.join(df_vars)
except Exception:
    pass

# Determine the actual output variable:
# 1. Prefer subtask_out (for subtask code)
# 2. Fall back to result (for merge code)
# 3. Fall back to the LAST newly-defined variable (if any)
# 4. Default to empty list
main_result = None
if 'subtask_out' in globals():
    main_result = globals()['subtask_out']
elif result is not None:
    main_result = result
elif new_keys:
    # Use the last variable defined by the user's code as fallback
    last_var = sorted(new_keys)[-1]
    last_val = globals()[last_var]
    # Auto-convert DataFrame to list of tuples
    if isinstance(last_val, pd.DataFrame):
        main_result = list(last_val.itertuples(index=False, name=None))
    elif isinstance(last_val, pd.Series):
        main_result = [[v] for v in last_val.tolist()]
    else:
        main_result = last_val

if main_result is None:
    main_result = []

with open(result_file_path, 'wb') as file:
    pickle.dump((main_result, df_vars_str), file)
    file.close()
"""


class CodeExecutor:
    """
    Sandboxed code executor for Python/Pandas code.

    Features:
    - Automatic table loading from SQLite database as DataFrames
    - Subprocess isolation via tempfile
    - Timeout control
    - Traceback capture
    - Structured result output

    Usage:
        executor = CodeExecutor()
        result = executor.execute(
            code="result = frpm[frpm['County Name'] == 'Alameda'][['Free_Rate']].max()",
            context={"db_path": "path/to/db.sqlite", "db_id": "california_schools"}
        )

        if result.success:
            print(f"Result: {result.result}")
        else:
            print(f"Error: {result.error}")
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        max_cpu_seconds: Optional[int] = None,
    ):
        """
        Initialize code executor.

        Args:
            timeout: Execution timeout in seconds
            max_memory_mb: Memory limit in megabytes
            top_k_row: Number of rows to load per table (None for all rows)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds or timeout
        self.logger = logger

    def execute(
        self,
        code: str,
        context: Optional[dict] = None,
        use_full_code: bool = False,
        top_k_row: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute Python code with given context.

        Args:
            code: Python code to execute
            context: Execution context containing:
                - db_path: Path to SQLite database
                - db_id: Database identifier
                - kb_type: Knowledge base type ('db', 'kg', 'table')
            use_full_code: Whether to use the full accumulated code
            top_k_row: Override for number of rows to load per table.
                       None means use all rows; if not provided, falls back to self.top_k_row.

        Returns:
            ExecutionResult with success status and result/error
        """
        start_time = time.time()
        context = context or {}

        try:
            validate_generated_code(code)

            # Step 1: Load database tables as DataFrames
            tables = self._load_tables(context, top_k_row=top_k_row)

            # Step 2: Wrap code with traceback template
            wrapped_code = self._wrap_code_with_traceback(code, use_full_code=use_full_code)

            # Step 3: Execute in isolated subprocess
            result, has_error, df_vars_str = self._execute_with_tempfile(tables, wrapped_code)

            execution_time = time.time() - start_time

            if has_error:
                return ExecutionResult(
                    success=False,
                    result=None,
                    error=result[0] if isinstance(result, list) and len(result) > 0 else str(result),
                    execution_time=execution_time,
                    df_vars=df_vars_str,
                )
            else:
                # Normalize result
                normalized_result = self._normalize_result(result)
                is_empty = len(normalized_result) == 0 if isinstance(normalized_result, list) else False
                return ExecutionResult(
                    success=True,
                    result=normalized_result,
                    error=None,
                    execution_time=execution_time,
                    df_vars=df_vars_str,
                    is_empty=is_empty,
                )

        except SecurityViolation as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"SecurityViolation: {e}",
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                result=None,
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return ExecutionResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
            )

    def _load_tables(self, context: dict, top_k_row: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load database tables as DataFrames based on context.

        Args:
            context: Execution context with db_path, db_id, kb_type
            top_k_row: Number of rows to load per table (None for all rows)

        Returns:
            Dictionary mapping table names to DataFrames
        """
        db_path = context.get("db_path")
        db_id = context.get("db_id", "unknown")
        kb_type = context.get("kb_type", "db")

        if kb_type == "multi":
            return self._load_multi_source_tables(context.get("sources", []), top_k_row)

        # For TableQA — table is already in context as table_df
        if kb_type == "table":
            return self._load_tableqa_tables(context, top_k_row=top_k_row)

        # For KG (GrailQA) — load from kg_dir, no SQLite DB needed
        if kb_type == "kg":
            kg_dir = context.get("kg_dir", db_path)
            csv_to_schema_map = context.get("csv_to_schema_map", {})
            return self._load_kg_tables(kg_dir, top_k_row=top_k_row, csv_to_schema_map=csv_to_schema_map)

        # For DB (NL2SQL) — requires SQLite database
        if not db_path or not os.path.exists(db_path):
            self.logger.warning(f"Database path not found: {db_path}")
            return {}

        if kb_type == "db":
            return self._load_sqlite_tables(db_path, top_k_row=top_k_row)

        self.logger.warning(f"Unknown kb_type: {kb_type}")
        return {}

    def _load_multi_source_tables(
        self, sources: list[dict], top_k_row: Optional[int]
    ) -> Dict[str, pd.DataFrame]:
        """Load heterogeneous sources into one collision-safe BOX namespace."""
        tables: Dict[str, pd.DataFrame] = {}
        for source in sources:
            kind = source.get("kind")
            prefix = source.get("prefix", "")
            loaded: Dict[str, pd.DataFrame]
            if kind == "db":
                loaded = self._load_sqlite_tables(source["path"], top_k_row)
            elif kind == "kg":
                loaded = self._load_kg_tables(source["kg_dir"], top_k_row)
                loaded = {name.replace(".", "_"): frame for name, frame in loaded.items()}
            elif kind == "table":
                frame = pd.read_csv(source["path"], nrows=top_k_row)
                loaded = {source.get("table_name", Path(source["path"]).stem): frame}
            else:
                raise ValueError(f"Unknown source kind: {kind}")

            for name, frame in loaded.items():
                target_name = f"{prefix}{name}" if prefix else name
                if target_name in tables:
                    raise ValueError(f"Duplicate BOX name across sources: {target_name}")
                tables[target_name] = frame
        return tables

    def _load_tableqa_tables(self, context: dict, top_k_row: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load table from context for TableQA tasks."""
        table_df = context.get("table_df")
        if table_df is None or table_df.empty:
            self.logger.warning("No table_df found in context for TableQA")
            return {}

        # Use all rows for TableQA (top_k_row doesn't make sense for small tables)
        tables = {"table": table_df}
        self.logger.info(f"Loaded TableQA table: {table_df.shape[0]} rows, {table_df.shape[1]} columns")
        return tables

    def _load_sqlite_tables(self, db_path: str, top_k_row: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load all tables from SQLite database with type normalization."""
        tables = {}
        try:
            conn = sqlite3.connect(db_path)
            # Fix non-UTF-8 encoded text in some BIRD/Spider databases
            # (e.g., wta_1.sqlite has Latin-1 text that fails UTF-8 decode)
            conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                query = f'SELECT * FROM "{table_name}"'
                if top_k_row is not None:
                    query += f" LIMIT {top_k_row}"

                df = pd.read_sql_query(query, conn)

                # Critical fix: Normalize CDSCode to string for proper JOIN operations
                # In BIRD dataset, CDSCode may be int in one table and string in another
                if 'CDSCode' in df.columns:
                    df['CDSCode'] = df['CDSCode'].astype(str)

                # Do NOT convert 'id'/'ID' to str — it breaks numeric comparisons
                # in LLM-generated code (e.g. df[df['id'] == 38] would fail)

                tables[table_name] = df

            conn.close()
            # self.logger.info(f"Loaded {len(tables)} tables from {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to load tables from {db_path}: {e}")

        return tables

    def _load_kg_tables(self, db_dir: str, top_k_row: Optional[int] = None, csv_to_schema_map: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """Load Knowledge Graph tables from CSV files.

        Args:
            db_dir: Directory containing CSV files
            top_k_row: Number of rows to load per table (None for all rows)
            csv_to_schema_map: Optional mapping from CSV file path to schema table name.
                              If provided, tables are named according to box_schema names
                              instead of CSV filenames.
        """
        tables = {}
        csv_to_schema_map = csv_to_schema_map or {}
        try:
            for filename in os.listdir(db_dir):
                if filename.lower().endswith(".csv"):
                    file_path = os.path.join(db_dir, filename)
                    # Use mapped schema name if available, fallback to stem
                    if file_path in csv_to_schema_map:
                        table_name = csv_to_schema_map[file_path]
                    else:
                        table_name = os.path.splitext(filename)[0]

                    if top_k_row is None:
                        df = pd.read_csv(file_path, dtype=str)
                    else:
                        df = pd.read_csv(file_path, dtype=str, nrows=top_k_row)

                    # Skip empty tables — they provide no data for queries
                    if df.empty:
                        continue

                    tables[table_name] = df

            self.logger.info(f"Loaded {len(tables)} KG tables from {db_dir}")
        except Exception as e:
            self.logger.error(f"Failed to load KG tables from {db_dir}: {e}")

        return tables

    def _load_single_table(self, table_path: str, top_k_row: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load a single table file."""
        try:
            table_name = os.path.splitext(os.path.basename(table_path))[0]

            if table_path.endswith('.csv'):
                df = pd.read_csv(table_path, nrows=top_k_row)
            elif table_path.endswith('.xlsx'):
                df = pd.read_excel(table_path, nrows=top_k_row)
            else:
                self.logger.warning(f"Unsupported table format: {table_path}")
                return {}

            self.logger.info(f"Loaded table {table_name} from {table_path}")
            return {table_name: df}

        except Exception as e:
            self.logger.error(f"Failed to load table from {table_path}: {e}")
            return {}

    def _wrap_code_with_traceback(self, code: str, use_full_code: bool = False) -> str:
        """
        Wrap code with traceback-capturing template.

        Args:
            code: Python code to wrap

        Returns:
            Wrapped code string
        """
        # Ensure pandas is imported
        if "import pandas as pd" not in code:
            code = "import pandas as pd\n" + code

        if not use_full_code and "result" not in code:
            code += "\nresult = [[1]]\n"

        # Indent code for try block (4 spaces per line)
        lines = code.split("\n")
        indented_lines = ["    " + line if line.strip() else line for line in lines]
        indented_code = "\n".join(indented_lines)

        # Replace placeholder in template
        wrapped_code = DEFAULT_PROMPT_FOR_TRACEBACK_CODE.replace("{{main_code}}", indented_code)

        return wrapped_code

    def _execute_with_tempfile(
        self,
        tables: Dict[str, pd.DataFrame],
        code: str,
    ) -> Tuple[Any, bool, str]:
        """
        Execute code in isolated subprocess with table data.

        Args:
            tables: Dictionary of table DataFrames
            code: Wrapped Python code with traceback template

        Returns:
            Tuple of (result, has_error, df_vars_str)
            - result: Main execution result or error info
            - has_error: Whether a runtime error occurred
            - df_vars_str: Description of newly defined DataFrame variables
        """
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as df_file:
            pickle.dump(tables, df_file)
            tables_file = df_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as code_file:
            code_file.write(code.encode("utf-8"))
            code_file_path = code_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as result_file:
            result_file_path = result_file.name

        try:
            # Execute subprocess
            sandbox_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
            }
            result = subprocess.run(
                [sys.executable, "-I", code_file_path, tables_file, result_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(code_file_path),
                env=sandbox_env,
                preexec_fn=self._apply_resource_limits if os.name == "posix" else None,
            )

            # Check for execution errors
            if result.returncode != 0:
                return self._handle_subprocess_error(result.stderr), True, ""

            # Deserialize result — now a (result, df_vars_str) tuple
            try:
                with open(result_file_path, "rb") as f:
                    output = pickle.load(f)
            except Exception as e:
                return [f"Failed to deserialize result: {str(e)}"], True, ""

            # Extract result and df_vars from the tuple
            if isinstance(output, tuple) and len(output) == 2:
                main_result, df_vars_str = output
            else:
                # Fallback for backward compatibility
                main_result, df_vars_str = output, ""

            # Check for runtime errors
            if self._is_execution_error(main_result):
                return main_result, True, df_vars_str

            return main_result, False, df_vars_str

        finally:
            # Clean up temporary files
            self._cleanup_temp_files(tables_file, code_file_path, result_file_path)

    def _apply_resource_limits(self) -> None:
        """Apply hard CPU and address-space limits inside the child process."""
        try:
            import resource

            cpu_limit = max(1, int(self.max_cpu_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            if self.max_memory_mb and self.max_memory_mb > 0:
                memory_bytes = int(self.max_memory_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ImportError, OSError, ValueError):
            # Timeout enforcement remains active on platforms lacking a limit.
            return

    def _is_execution_error(self, output: Any) -> bool:
        """Check if output indicates an execution error."""
        try:
            if not isinstance(output, (list, tuple)) or len(output) == 0:
                return False
            first_elem = output[0]
            if not isinstance(first_elem, str):
                return False
            return "Execution failed" in first_elem or "Traceback" in first_elem
        except Exception:
            return False

    def _handle_subprocess_error(self, stderr: str) -> list:
        """Format subprocess execution errors."""
        traceback_lines = stderr.split("\n")
        last_line = traceback_lines[-2] if len(traceback_lines) > 1 else stderr
        summary = f"Execution failed: {last_line}"
        full_traceback = f"Traceback (most recent call last):\n{stderr}"
        return [[full_traceback, summary]]

    def _cleanup_temp_files(self, *file_paths: str) -> None:
        """Remove temporary files."""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                self.logger.warning(f"Failed to remove temp file '{path}': {e}")

    def _normalize_result(self, result: Any) -> list:
        """
        Normalize execution result to list of tuples format.

        Also normalizes NaN values to None for consistent downstream handling.

        Args:
            result: Raw execution result

        Returns:
            Normalized list of tuples
        """
        _SCALAR_TYPES = (int, float, str, type(None))

        # Handle zip objects
        if isinstance(result, zip):
            return list(result)

        # Handle scalar outputs
        if isinstance(result, _SCALAR_TYPES):
            return [(self._sanitize_value(result),)]

        # Handle iterables
        if isinstance(result, (list, tuple)):
            try:
                if len(result) == 0:
                    return result
                # Wrap scalar elements as tuples
                return [
                    (self._sanitize_value(elem),) if isinstance(elem, _SCALAR_TYPES)
                    else tuple(self._sanitize_value(v) for v in elem)
                    for elem in result
                ]
            except Exception:
                return [result]

        # Default
        return [result]

    @staticmethod
    def _sanitize_value(val: Any) -> Any:
        """Convert NaN (float or numpy) to None for consistent downstream handling."""
        if val is None:
            return None
        # Check for float NaN
        if isinstance(val, float) and val != val:  # NaN check: NaN != NaN
            return None
        # Check for numpy NaN
        try:
            import numpy as np
            if isinstance(val, (np.floating, np.integer)) and np.isnan(val):
                return None
            if isinstance(val, np.ndarray) and val.size == 1 and np.isnan(val).item():
                return None
        except (ImportError, TypeError, ValueError):
            pass
        return val


# Convenience function
def execute_pandas_code(
    code: str,
    db_path: str,
    context: Optional[dict] = None,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Convenience function to execute Pandas code.

    Args:
        code: Python Pandas code
        db_path: Path to SQLite database
        context: Additional context variables
        timeout: Execution timeout

    Returns:
        ExecutionResult
    """
    executor = CodeExecutor(timeout=timeout)
    return executor.execute(code, {"db_path": db_path, **(context or {})})
