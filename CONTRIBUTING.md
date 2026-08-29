# Contributing

Open an issue before substantial changes so method-level changes can be kept
separate from faithful implementation fixes. Pull requests should be focused,
document behavioral changes, and include synthetic tests that run without
benchmark data or external API calls.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not integration"
python -m compileall -q core datasets models prompts utils scripts run.py
python scripts/audit_repository.py --mode code-only --strict
python -m build
```

Do not commit benchmark samples, memory, few-shot artifacts, model files,
results, logs, credentials, private endpoints, manuscript review material, or
machine-specific absolute paths. By contributing, you agree that your changes
are provided under Apache-2.0.
