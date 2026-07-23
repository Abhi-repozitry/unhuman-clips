# Contributing

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment:
   ```bash
   python -m venv backend/venv
   source backend/venv/bin/activate  # Linux/Mac
   # or: backend\venv\Scripts\activate  # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   pip install ruff pre-commit pytest
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

Branch naming conventions:
- `feature/` — New features
- `fix/` — Bug fixes
- `refactor/` — Code refactoring
- `docs/` — Documentation changes
- `test/` — Adding or updating tests

### 2. Make Changes

- Follow the existing code style (see [Code Style](#code-style))
- Add tests for new functionality
- Update documentation if adding user-facing features
- Keep commits focused and well-described

### 3. Run Checks

```bash
# Lint
ruff check backend/ tests/

# Format
ruff format backend/ tests/

# Tests
pytest tests/ -m "not integration"

# Smoke test
python scripts/smoke_test.py
```

All checks must pass before submitting a PR. Pre-commit hooks run these automatically.

### 4. Submit a Pull Request

- Fill out the PR template
- Reference any related issues
- Include screenshots/video for UI changes
- Ensure CI passes

---

## Code Style

### Python

- **Formatter/Linter**: Ruff (replaces flake8, isort, black)
- **Line length**: 88 characters (default)
- **Import order**: isort-compatible (handled by Ruff)
- **Type hints**: Use `from __future__ import annotations` at the top of every file

```python
"""Module docstring."""
from __future__ import annotations

import logging
from typing import Any

__all__ = ["my_function"]
logger = logging.getLogger(__name__)
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Functions | `snake_case` | `process_group()` |
| Classes | `PascalCase` | `GroupOrchestrator` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_WORKERS` |
| Private | `_leading_underscore` | `_compute_group_count_target()` |
| Modules | `snake_case` | `narration_validator.py` |

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def my_function(input_data: list[dict], reporter: Any) -> list[dict]:
    """Process input data and return results.

    Args:
        input_data: List of input records from previous stage.
        reporter: ProgressReporter for status updates.

    Returns:
        Processed list of output records.

    Raises:
        ValueError: If input_data is empty.
    """
```

### Testing

- **Framework**: pytest
- **Mocking**: `unittest.mock` (MagicMock for sync, AsyncMock for async)
- **Pattern**: Arrange → Act → Assert
- **One assertion per concept**: Keep tests focused
- **Naming**: `test_<what>_<condition>_<expected>()`

```python
class TestMyFunction:
    def test_empty_input_returns_empty(self, mock_reporter):
        """Given empty input, function should return empty list."""
        result = my_function([], mock_reporter)
        assert result == []

    def test_valid_input_processes_correctly(self, mock_reporter):
        """Given valid input, function should process and return results."""
        input_data = [{"key": "value"}]
        result = my_function(input_data, mock_reporter)
        assert len(result) == 1
        assert result[0]["processed"] is True
```

---

## Commit Messages

Use conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` — New feature
- `fix` — Bug fix
- `refactor` — Code refactoring (no behavior change)
- `test` — Adding/updating tests
- `docs` — Documentation
- `chore` — Maintenance (deps, CI, etc.)

**Examples:**

```
feat(pipeline): add VAD-driven audio ducking
fix(clipper): handle timeout on slow NVENC encoding
refactor(queue_manager): extract group orchestration logic
test(analyzer): add edge cases for JSON repair
docs: update README with architecture diagram
```

---

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass (`pytest tests/`)
- [ ] Lint passes (`ruff check backend/ tests/`)
- [ ] Format passes (`ruff format backend/ tests/ --check`)
- [ ] No new type errors (if type checking is configured)
- [ ] Smoke test passes (`python scripts/smoke_test.py`)
- [ ] Documentation updated (if applicable)

### PR Description

Include:
1. **What** — Brief description of the change
2. **Why** — Motivation and context
3. **How** — Implementation approach (if non-obvious)
4. **Testing** — How you verified the change
5. **Screenshots** — For UI changes

### Review Process

- At least 1 approval required for merge
- Address all review comments
- Squash commits before merging (clean history)

---

## Project Structure

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full code structure and architecture.

Key files to understand:
- `backend/queue_manager.py` — Job queue and pipeline entry point
- `backend/pipeline/orchestrator.py` — Per-group pipeline stages
- `backend/pipeline/compositor.py` — Video composition and ffmpeg filter chains
- `backend/pipeline/analyzer.py` — LLM interaction and reel plan generation
- `backend/models.py` — All Pydantic data models

---

## Adding Dependencies

1. Add to `backend/requirements.txt` with version pin:
   ```
   package-name>=1.0,<2.0
   ```
2. Run `pip install -r backend/requirements.txt` to verify
3. Commit both files

---

## Reporting Issues

Use GitHub Issues. Include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, GPU, Python version)
- Relevant log output

---

## License

By contributing, you agree that your contributions will be licensed under the project's private license.
