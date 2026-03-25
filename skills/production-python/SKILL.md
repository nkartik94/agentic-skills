---
name: production-python
description: Production-grade Python coding conventions for modules, classes, functions, Pydantic models, SQLAlchemy models, and tests. Use this skill when writing new Python files, modifying existing Python code, creating data models, setting up project structure, or reviewing Python code for best practices. Triggers on any task involving Python source files (.py), test files, notebooks, or project scaffolding.
license: MIT
metadata:
  author: nkartik94
  version: "1.2.0"
---

# Production Python

Apply every rule below whenever writing or modifying Python code.

## When to Apply

- Writing new Python modules, classes, or functions
- Creating Pydantic models or SQLAlchemy ORM models
- Setting up project structure (imports, logging, config, paths)
- Writing or updating tests
- Reviewing Python code for production readiness
- Scaffolding FastAPI endpoints, repositories, or pipelines

---

## Quick Reference

| Rule | Pattern |
|------|---------|
| Module header | `"""Module Name: filename.py\nDescription: ..."""` |
| Section markers | 77-dash lines with `# SECTION: Name` |
| End of module | `# END OF MODULE` block — always last |
| Imports | 3 groups: stdlib / third-party / local, alphabetical |
| Logger | `logger = setup_logger(__name__)` — never `print()` |
| Type hints | All args + return, `Optional[X]` not `X \| None` |
| Docstrings | Google style, class docstring on class not `__init__` |
| Naming | `snake_case` funcs, `PascalCase` classes, `UPPER_SNAKE` constants |
| Formatting | 88-char target (Black/Ruff default), trailing commas, 2 blanks between top-level |
| Strings | f-strings only, `pathlib.Path` for all paths |
| Errors | Specific exceptions, `from e` chaining, log before raise |
| Testing | `test_<func>_<scenario>`, Arrange/Act/Assert, mock all I/O |
| Architecture | Repository pattern, centralized paths, registry for dispatch |
| Pydantic | `ConfigDict`, `Field(description=...)`, separate Create/Response |
| SQLAlchemy | `Mapped[T]` + `mapped_column()`, `back_populates` |
| Package mgmt | `uv add`, `uv sync --frozen` in CI |

---

## 1. Module Header

Every `.py` file starts with a module docstring — no exceptions:

```python
"""
Module Name: filename.py

Description:
    Clear, concise description of the module's purpose and role
    in the overall system. Can span multiple lines.
"""
```

No author, date, or version — that belongs in version control.

---

## 2. Section Markers

Wrap every logical section in 77-character dash lines with blank lines before and after:

```python
# --------------------------------------------------------------------------
# SECTION: Imports
# --------------------------------------------------------------------------
```

Standard order (include only what's needed): `Imports` -> `Logger Initialization` -> `Constants` -> `Type Aliases` -> [content sections]

**Class sub-sections** use shorter inline markers:

```python
class ExampleClass:
    # --- Constructor ---
    def __init__(self, param: str) -> None: ...
    # --- Public API ---
    def process(self, data: List[Dict]) -> pd.DataFrame: ...
    # --- Private Helpers ---
    def _validate(self) -> None: ...
```

---

## 3. End-of-Module Marker

The very last thing in every `.py` file — no code or comments after it:

```python
# --------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------
```

---

## 4. Import Organization

Three groups with comment headers, separated by blank lines:

```python
# Standard library imports
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd
from pydantic import BaseModel

# Local application imports
from src.logger import setup_logger
from src.utils.helpers import load_prompt
```

Rules:
- Alphabetical within each group — bare `import X` before `from X import Y`
- Group multiple from same module: `from typing import Dict, List, Optional`
- Absolute imports only — never relative (`from .module import x`), **except** `__init__.py` re-exports where relative imports are acceptable per PEP 8
- Never `import *`

---

## 5. Logger

Every project needs a central `logger.py` (see [references/REFERENCE.md](references/REFERENCE.md) for full implementation). In every module, one line after imports:

```python
logger = setup_logger(__name__)
```

**Log level guide:**

```python
logger.debug(f"Chunk {i}/{total} processed — tokens: {tokens}")
logger.info(f"Processing complete — {count} records in {elapsed:.2f}s")
logger.warning(f"Unexpected mode '{mode}', falling back to default")
logger.error(f"Chain execution failed: {e}")
logger.critical("Database connection lost — shutting down")
```

Never use `print()` in production code.

**Exception:** `%`-style formatting is acceptable in `logger.*()` calls (lazy string evaluation — skips construction if the message is filtered):

```python
logger.info("Processed %d records in %.2fs", count, elapsed)
```

---

## 6. Type Hints

Required on all function signatures — no exceptions for public APIs:

```python
def load_prompt(file_path: str) -> str: ...
def process(items: List[str], limit: int = 10) -> Optional[Dict[str, int]]: ...
```

Multi-line signatures — one param per line, trailing comma:

```python
def process_chunks(
    chunks: List[Dict[str, Any]],
    prompts: List[Dict[str, str]],
    run_sequentially: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int]:
```

Class attributes — document in the class docstring `Attributes:` section.

Rules:
- `Optional[X]` not `X | None` (Python 3.9 compat)
- Import `Dict`, `List`, `Tuple`, `Optional` from `typing`
- Always annotate return type — even `-> None`
- Use `Any` sparingly

---

## 7. Docstrings — Google Style

**Class docstring** — on the class, never on `__init__`:

```python
class BaseAgent:
    """
    Base class providing shared methods for all LLM-based agents.

    Attributes:
        model_name (str): LLM model identifier.
        response_schema: Pydantic model for parsing output.
    """
```

**Function docstring:**

```python
def fetch_data(source: str, timeout: int = 30) -> List[Dict]:
    """
    Fetch structured data from source.

    Args:
        source: URL or file path to fetch from.
        timeout: Request timeout in seconds (default: 30).

    Returns:
        List of record dicts. Empty list if nothing found.

    Raises:
        ValueError: If source is empty or malformed.
    """
```

Pydantic fields — always use `Field(description=...)`:

```python
row_index: int = Field(description="Row index in the original data.")
confidence: float = Field(description="Confidence score 0.0-1.0.", default=1.0)
```

---

## 8. Naming Conventions

| Kind | Convention | Example |
|------|------------|---------|
| Variables, functions, modules | `snake_case` | `user_id`, `parse_resume()` |
| Private methods / attributes | `_prefix` | `_validate()`, `_cache` |
| Classes | `PascalCase` | `MatchResult`, `BaseAgent` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Type aliases | `PascalCase` | `ChunkType = Dict[str, Any]` |

Descriptive, unambiguous names. Avoid abbreviations except standard ones (`url`, `id`, `db`, `api`).

Class patterns: `[Domain]Agent`, `[Domain]Manager`, `[Domain]Repository`, descriptive nouns for models.

---

## 9. Formatting Standards

Target line length: **88 characters** (Black/Ruff default). Hard limit: 100.

**Blank lines:** 2 between top-level definitions, 1 between methods, 1 before/after section markers.

**Long signatures** — one param per line, trailing comma:

```python
def process_data(
    input_path: Path,
    output_dir: Path,
    chunk_size: int = 1000,
) -> List[Dict[str, Any]]:
```

**String continuation** — parenthesized implicit concatenation:

```python
error_message = (
    f"Failed to process file {file_path}. "
    f"Error: {error_details}."
)
```

**Method chains** — one call per line:

```python
result = (
    df.query("status == 'active'")
    .groupby("category")
    .agg(total=("amount", "sum"))
    .reset_index()
)
```

**Comments:** 2 spaces before `#` for inline; block comments above code preferred.

**Trailing newline:** Every file ends with exactly one newline character. Enable in VSCode: `"files.insertFinalNewline": true`. Ruff auto-fixes `W292`.

---

## 10. Strings & Paths

**f-strings exclusively** — never `.format()` or `%`:

```python
logger.info(f"Processing {count} records for job {job_id}")
```

**`pathlib.Path` always** — never string concatenation:

```python
output_path = Path(base_dir) / "results" / f"{job_id}.json"
```

---

## 11. Error Handling

Catch specific exceptions. Never bare `except:`. Log before raising:

```python
try:
    result = parse_document(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    raise
except ValueError as e:
    logger.warning(f"Parse failed for {file_path}: {e}")
    return None
```

**Exception chaining** — always `from e` to preserve traceback:

```python
except json.JSONDecodeError as e:
    raise ExtractionError(f"Invalid JSON in {file_path}") from e
```

**`finally` for cleanup:**

```python
try:
    conn = get_connection()
    result = conn.execute(query)
finally:
    conn.close()
```

**Retry with Tenacity** for transient failures:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def call_llm(prompt: str) -> str:
    """Call LLM API with automatic retry on transient failures."""
    return client.chat(prompt).content
```

**Full traceback logging:** `logger.error(f"Failed: {e}\n{traceback.format_exc()}")`

Custom exceptions for domain errors: `class ExtractionError(Exception): ...`

---

## 12. Architecture Patterns

**Repository Pattern** — all DB access through repository classes:

```python
class JobRepository:
    """Encapsulates all job-related database operations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, job_id: uuid.UUID) -> Optional[Job]:
        return self.session.get(Job, job_id)

    def create(self, **kwargs) -> Job:
        job = Job(**kwargs)
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job
```

**Centralized Paths** — single source of truth in `config/paths.py`:

```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
```

**Registry Pattern** — extensible dispatch:

```python
PROCESSOR_REGISTRY: Dict[str, Callable] = {}

def register_processor(name: str, func: Callable) -> None:
    PROCESSOR_REGISTRY[name] = func

def process(name: str, data: Dict) -> Dict:
    return PROCESSOR_REGISTRY.get(name, process_default)(data)
```

**Configuration** — YAML in `config/`, never hardcoded values.

---

## 13. Environment Variables

Use `pydantic-settings` to load and validate all configuration from environment variables:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    api_key: str
    debug: bool = False
    max_workers: int = 4

settings = Settings()
```

Rules:
- Never hardcode secrets or environment-specific values — use env vars
- Provide a `.env.example` with all required keys (no real values)
- Validate at startup so missing config fails fast, not at runtime
- Import the singleton `settings` object, never call `os.getenv()` in business logic

Full implementation in [references/REFERENCE.md](references/REFERENCE.md).

---

## 14. Async Patterns

Use `async def` / `await` for I/O-bound work (HTTP, DB, file I/O). Never block the event loop with synchronous calls:

```python
import asyncio
from typing import List

async def fetch_all(urls: List[str]) -> List[str]:
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_one(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch a single URL."""
    async with session.get(url) as response:
        return await response.text()
```

Rules:
- `async def` for any function that calls `await` — never mix sync/async carelessly
- Use `asyncio.gather()` for concurrent independent tasks
- Use `asyncio.run()` at the entry point — never `loop.run_until_complete()`
- Wrap sync blocking calls with `asyncio.to_thread()` if unavoidable
- Same logging, type hints, docstring, and error handling rules apply

---

## 15. TypedDict & Protocol

**`TypedDict`** — typed dict shape without Pydantic overhead (no validation, pure annotation):

```python
from typing import TypedDict

class JobConfig(TypedDict):
    model_name: str
    chunk_size: int
    dry_run: bool
```

**`Protocol`** — structural typing (duck typing with type safety):

```python
from typing import Protocol

class Runnable(Protocol):
    def run(self, data: dict) -> dict: ...
    def validate(self) -> bool: ...

def execute(runner: Runnable) -> dict:
    """Accepts any object implementing Runnable — no inheritance required."""
    if not runner.validate():
        raise ValueError("Validation failed")
    return runner.run({})
```

Use `TypedDict` over plain `Dict[str, Any]` when the shape is known. Use `Protocol` over `ABC` when you want structural typing without forcing inheritance.

---

## 16. Canonical Log Lines

For API endpoints and pipeline runs, emit **one structured summary line** at operation end that collates all key telemetry:

```python
import json
import time
from typing import Any, Dict

class CanonicalLog:
    """Accumulates fields and emits one structured log line per operation."""

    def __init__(self) -> None:
        self._fields: Dict[str, Any] = {}
        self._start = time.monotonic()

    def set(self, **kwargs: Any) -> None:
        """Add or update fields."""
        self._fields.update(kwargs)

    def emit(self, logger) -> None:
        """Emit the canonical line with auto-computed duration."""
        self._fields["duration_ms"] = round((time.monotonic() - self._start) * 1000)
        logger.info("CANONICAL %s", json.dumps(self._fields))
```

**Usage pattern — always emit in `finally`:**

```python
def handle_request(request):
    log = CanonicalLog()
    log.set(event="extraction.run", path=request.url.path)
    try:
        result = run_extraction(request)
        log.set(status="success", rows=len(result), tokens_in=result.input_tokens)
        return result
    except Exception as e:
        log.set(status="error", error=str(e))
        raise
    finally:
        log.emit(logger)  # Always fires — even on exception
```

**Output:**
```
CANONICAL {"event": "extraction.run", "path": "/extract", "status": "success", "rows": 42, "duration_ms": 820}
```

Full implementation with request ID correlation and JSON logging mode in [references/REFERENCE.md](references/REFERENCE.md).

---

## 17. Testing Patterns

**Naming:** `test_<module>.py` in `tests/` mirroring `src/`. Functions: `test_<func>_<scenario>`.

**Structure** — Arrange / Act / Assert:

```python
def test_process_chunks_returns_expected_count() -> None:
    """Verify process_chunks returns one result per input chunk."""
    # Arrange
    chunks = [{"text": "hello"}, {"text": "world"}]
    pipeline = ProcessingPipeline(model_name="test")

    # Act
    results = pipeline.process_chunks(chunks)

    # Assert
    assert len(results) == 2
    assert all("output" in r for r in results)
```

**Fixtures** — `@pytest.fixture` for shared setup, `conftest.py` for cross-module:

```python
@pytest.fixture
def db_session(tmp_path: Path) -> Generator[Session, None, None]:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
```

**Mocking** — prefer `monkeypatch` over `unittest.mock.patch`:

```python
def test_fetch_data_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", Mock(side_effect=Timeout))
    assert fetch_data("https://api.example.com") is None
```

Rules: one behavior per test, `pytest.raises` for expected exceptions, mock all I/O.

---

## 18. Pydantic Models

**Separate models for API boundaries:**

```python
class CandidateCreate(BaseModel):
    """Request schema for creating a candidate."""
    name: str = Field(description="Full legal name.")
    email: str = Field(description="Primary email address.")

class CandidateResponse(BaseModel):
    """Response schema — never expose ORM objects directly."""
    candidate_id: uuid.UUID
    name: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
```

**Structured LLM output:**

```python
class ExtractedField(BaseModel):
    field_name: str = Field(description="Name of the extracted field.")
    value: str = Field(description="Extracted value.")
    confidence: float = Field(description="Confidence score 0.0-1.0.")

class ExtractionResult(BaseModel):
    fields: List[ExtractedField]
```

Rules: `ConfigDict` (not deprecated `class Config`), `Field(description=...)` on all fields, one model per concept.

---

## 19. SQLAlchemy Models (2.0)

```python
class Job(Base):
    """ORM model for the jobs table."""
    __tablename__ = "jobs"

    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    candidates: Mapped[List["Candidate"]] = relationship(back_populates="job")

    def __repr__(self) -> str:
        return f"Job(job_id={self.job_id!r}, title={self.title!r})"
```

Rules:
- `Mapped[T]` + `mapped_column()` — never legacy `Column()`
- UUID primary keys, all timestamps `timezone=True`, `server_default=func.now()`
- `back_populates` (never `backref`), `__repr__` on every model
- No business logic in models — pure data containers
- Never `max()` on UUID columns — use `row_number()` window functions

---

## 20. Anti-Patterns

```python
try: ...
except: ...                        # Bare except — hides bugs

print("done")                      # Use logger, never print()

def add(items=[]):                 # Mutable default — shared state bug
    items.append(1)

from module import *               # Wildcard — pollutes namespace

path = "/base/" + subdir + "/f.txt"  # Use pathlib

msg = "Hello %s" % name            # Use f-strings

from .utils import helper          # Use absolute imports

API_KEY = "sk-abc123"              # Use environment variables
```

Also avoid: logic in `__init__.py` (re-exports only), god functions (> ~50 lines).

---

## 21. Pre-Commit Checklist

### Structure & Organization
- [ ] Module docstring with `Module Name:` and `Description:`
- [ ] Section markers (77-dash lines) around logical sections
- [ ] `# END OF MODULE` as the very last line
- [ ] Imports in 3 groups, sorted alphabetically
- [ ] `logger = setup_logger(__name__)` present
- [ ] Blank lines follow PEP 8 (2 between top-level, 1 between methods)

### Documentation
- [ ] Google-style docstrings on all public functions and classes
- [ ] Class docstrings include an `Attributes:` section
- [ ] Function docstrings include `Args:`, `Returns:`, and `Raises:` where applicable
- [ ] Complex logic has explanatory inline comments
- [ ] Class subsection headers (`# --- Public API ---` etc.) present in non-trivial classes

### Naming & Types
- [ ] Variables and functions use `snake_case`; private helpers prefixed with `_`
- [ ] Classes use `PascalCase`, constants use `UPPER_SNAKE_CASE`
- [ ] Type hints on all public function args + return values
- [ ] Type hint imports come from `typing` (e.g. `List`, `Optional`, `Dict`)

### Code Quality
- [ ] No `print()`, no bare `except:`, no mutable defaults
- [ ] Specific exception types in all `try/except` blocks; `from e` chaining
- [ ] Logging at appropriate levels (`debug`/`info`/`warning`/`error`)
- [ ] Input validation with descriptive error messages at entry points
- [ ] f-strings throughout, `pathlib.Path` for all file paths
- [ ] No relative imports (use `from src.` absolute paths; `__init__.py` re-exports are the only exception)
- [ ] File ends with exactly one trailing newline

### Patterns & Conventions
- [ ] Default parameters use module-level constants, not magic literals
- [ ] No hardcoded paths — use `config/paths.py` or `pathlib.Path` constants
- [ ] Retry decorators on external API / DB calls
- [ ] Repository pattern for all database access (if applicable)
- [ ] Pydantic models for all API request/response schemas (if applicable)

### Testing & Verification
- [ ] Public functions have at least one test
- [ ] No unresolved `TODO` or `FIXME` comments
- [ ] Line length within 88 chars

---

## 22. CHANGELOG.md

Follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) + [Semantic Versioning](https://semver.org/). Full template and example in [references/REFERENCE.md](references/REFERENCE.md).

Rules: reverse chronological, `YYYY-MM-DD` dates, omit empty sections. MAJOR = breaking, MINOR = features, PATCH = fixes.

Each release entry uses this structure:
- `## [VERSION] — YYYY-MM-DD` heading
- 1–2 sentence **summary paragraph** (plain English, what changed and why)
- Sections: `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`

| Section | Use for |
|---------|---------|
| **Added** | New features and capabilities |
| **Changed** | Modifications to existing features; mark breaking changes |
| **Deprecated** | Features marked for future removal; include migration guidance |
| **Removed** | Previously deprecated features now gone |
| **Fixed** | Bug fixes |
| **Security** | Vulnerabilities patched; include CVE numbers when applicable |

Best practices: mention file paths and function names in entries, explain *why* not just *what*, link to issues/PRs.

---

## 23. README.md

Required sections: **Title** -> **Stack** -> **Architecture** -> **Quick Start** -> **Environment Variables** -> **Development** -> **Project Structure** -> **Docs**

Full template in [references/REFERENCE.md](references/REFERENCE.md). Start broad (what + why), get specific (how). Quick Start < 5 steps.

---

## 24. Jupyter Notebooks

Notebooks in `notebooks/`, `snake_case` names (no dates). Cell order: title -> imports -> config -> processing -> results. Use `setup_logger(__name__)`. Clear outputs before committing. Full conventions in [references/REFERENCE.md](references/REFERENCE.md).

---

## 25. Package Management with `uv`

```bash
uv init                           # Initialize new project
uv add fastapi                    # Add dependency
uv add pandas==2.2.3              # Pin exact version for production
uv add --dev pytest ruff black    # Dev dependency
uv remove old-package             # Remove dependency
uv sync                           # Install from lock file
uv sync --dev                     # Install including dev deps
uv sync --no-dev --frozen         # CI/CD — production only, no lock update
uv lock --upgrade                 # Update all packages to latest compatible
uv lock --upgrade-package fastapi # Update single package
uv run pytest                     # Run command in project environment
```

Rules: pin exact versions in production, commit both `pyproject.toml` and `uv.lock`, dev tools under `[project.optional-dependencies]`.

Full guide — pyproject.toml structure, Docker workflow, CI/CD GitHub Actions snippet, and uv vs pip comparison: [references/REFERENCE.md](references/REFERENCE.md).

---

## 26. Docker

Dockerfiles follow the same section-marker convention as Python modules — wrap each logical block in labeled sections:

```dockerfile
# SECTION: Base Image
FROM python:3.12.11-slim

# SECTION: Python Dependencies Installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT="/usr/local"
RUN uv sync --no-dev --frozen

# SECTION: Application Code
COPY . .
```

Key pattern: install uv from its official image, set `UV_PROJECT_ENVIRONMENT` to install into system Python (no `.venv` in containers), then `uv sync --no-dev --frozen` for reproducible production installs.

Full Dockerfile template and Docker command reference (build, run, compose, debug): [references/REFERENCE.md](references/REFERENCE.md).

---

## 27. Common Patterns

### Optional progress callback

Accept an optional callback to report progress — lets callers wire in a UI (Streamlit, CLI progress bar) without coupling the function to any framework:

```python
from typing import Callable, Optional

def process_items(
    items: List[Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    for i, item in enumerate(items):
        result = process_single(item)
        if progress_callback:
            progress_callback(i + 1, len(items))  # (completed, total)
    return results
```

### Dynamic parameter dispatch with `inspect.signature`

When calling functions from a registry where different implementations accept different parameters, use `inspect.signature` to build kwargs dynamically rather than a chain of `if` checks:

```python
import inspect

def dispatch(func: Callable, available: Dict[str, Any]) -> Any:
    sig = inspect.signature(func)
    kwargs = {k: available[k] for k in sig.parameters if k in available}
    return func(**kwargs)
```

---

For full templates, complete code examples, env var setup, async patterns, canonical logging implementation, and additional resources, see [references/REFERENCE.md](references/REFERENCE.md).
