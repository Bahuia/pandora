import pandas as pd

from utils.code_executor import CodeExecutor


def test_table_execution_under_default_quotas():
    result = CodeExecutor(timeout=10).execute(
        "result = list(table[['x']].itertuples(index=False, name=None))",
        {"kb_type": "table", "table_df": pd.DataFrame({"x": [1, 2]})},
    )
    assert result.success
    assert result.result == [(1,), (2,)]


def test_disallowed_import_is_rejected_before_execution():
    result = CodeExecutor(timeout=10).execute("import os\nresult = []", {})
    assert not result.success
    assert "SecurityViolation" in result.error
