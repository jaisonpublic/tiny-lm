---
description: Always use uv and the project virtual environment for Python commands
---

## Rule: Always Use `uv` and the Virtual Environment

// turbo-all

**IMPORTANT**: Never use `pip`, `python`, or `pip install` directly. Always use `uv` to ensure the project's `.venv` virtual environment is used.

### Installing dependencies
```bash
# ✅ Correct
uv add <package-name>

# ❌ Wrong
pip install <package-name>
```

### Running Python scripts
```bash
# ✅ Correct
uv run python <script.py>

# ❌ Wrong
python <script.py>
```

### Running tools/CLIs
```bash
# ✅ Correct
uv run chainlit run app.py
uv run pytest

# ❌ Wrong
chainlit run app.py
pytest
```
