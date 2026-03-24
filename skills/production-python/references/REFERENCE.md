# Production Python — Extended Reference

Detailed templates, full implementations, and supplementary patterns for the production-python skill.
See [SKILL.md](../SKILL.md) for the core rules.

---

## Logger — Full `logger.py` Implementation

Drop this file into any project as `src/logger.py` (or adjust the path to match your layout):

```python
"""
Module Name: logger.py

Description:
    Centralized logging configuration.
    Rotating file handler + console handler with noise suppression.
"""

# --------------------------------------------------------------------------
# SECTION: Imports
# --------------------------------------------------------------------------

# Standard library imports
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# --------------------------------------------------------------------------
# SECTION: Setup Logger
# --------------------------------------------------------------------------

def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with rotating file and console handlers.

    Args:
        name: Logger name — always pass __name__ from the calling module.
        log_dir: Directory to write log files (created if missing).
        log_file: Log file name.
        max_bytes: Max file size before rotation. Default: 5 MB.
        backup_count: Number of rotated files to retain.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Suppress noisy third-party library logs
    for noisy_lib in ("urllib3", "asyncio", "langchain", "httpx", "httpcore"):
        logging.getLogger(noisy_lib).setLevel(logging.CRITICAL)

    logger.propagate = False
    return logger

# --------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------
```

---

## Environment Variables — `pydantic-settings` Setup

Install: `uv add pydantic-settings`

```python
"""
Module Name: settings.py

Description:
    Application settings loaded and validated from environment variables.
    Single source of truth for all configuration.
"""

# --------------------------------------------------------------------------
# SECTION: Imports
# --------------------------------------------------------------------------

# Standard library imports
from typing import Optional

# Third-party imports
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# SECTION: Settings
# --------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application settings — all values loaded from environment.

    Attributes:
        database_url (str): Full database connection string.
        api_key (str): External API key.
        debug (bool): Enable debug mode.
        max_workers (int): Thread pool size.
        log_format (str): "text" for local dev, "json" for production.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    database_url: str
    api_key: str
    debug: bool = False
    max_workers: int = 4
    log_format: str = "text"
    log_level: str = "INFO"
    app_env: str = "development"

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Ensure log format is a recognized value."""
        allowed = {"text", "json"}
        if v not in allowed:
            raise ValueError(f"log_format must be one of {allowed}")
        return v


# Module-level singleton — import this, never call os.getenv() in business logic
settings = Settings()

# --------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------
```

**`.env.example`** — commit this, never commit `.env`:

```dotenv
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
API_KEY=your-api-key-here
DEBUG=false
MAX_WORKERS=4
LOG_FORMAT=text
LOG_LEVEL=INFO
APP_ENV=development
```

**Usage:**
```python
from src.settings import settings

logger.info(f"Starting in {settings.app_env} mode")
conn = create_engine(settings.database_url)
```

---

## Canonical Log Lines — Full Implementation

Drop `src/core/logging/canonical.py` into any project:

```python
"""
Module Name: canonical.py

Description:
    Canonical log line context for structured per-request/operation telemetry.
    Emits one structured summary line per operation, enabling easy log querying.
"""

# --------------------------------------------------------------------------
# SECTION: Imports
# --------------------------------------------------------------------------

# Standard library imports
import json
import logging
import time
from typing import Any, Dict

# --------------------------------------------------------------------------
# SECTION: Canonical Log Context
# --------------------------------------------------------------------------

class CanonicalLog:
    """
    Accumulates key-value fields throughout an operation and emits
    one structured log line at the end.

    Usage:
        log = CanonicalLog()
        log.set(event="job.run", job_id=job_id)
        try:
            result = run_job()
            log.set(status="success", rows=len(result))
        except Exception as e:
            log.set(status="error", error=str(e))
            raise
        finally:
            log.emit(logger)
    """

    def __init__(self) -> None:
        self._fields: Dict[str, Any] = {}
        self._start: float = time.monotonic()

    def set(self, **kwargs: Any) -> None:
        """Add or update fields on the canonical line."""
        self._fields.update(kwargs)

    def emit(self, logger: logging.Logger) -> None:
        """Emit the canonical log line. Always call in a finally block."""
        self._fields["duration_ms"] = round(
            (time.monotonic() - self._start) * 1000
        )
        logger.info("CANONICAL %s", json.dumps(self._fields, default=str))

# --------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------
```

**Pattern — always emit in `finally`:**

```python
def run_pipeline(job_id: int, file_path: str) -> Dict:
    """Run extraction pipeline with canonical telemetry."""
    log = CanonicalLog()
    log.set(event="pipeline.run", job_id=job_id, file=file_path)

    try:
        result = extract(file_path)
        log.set(
            status="success",
            rows_extracted=len(result),
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )
        return result

    except Exception as e:
        log.set(status="error", error=str(e))
        raise

    finally:
        log.emit(logger)  # Fires even on exception
```

**Output:**
```
CANONICAL {"event": "pipeline.run", "job_id": 42, "file": "sov.xlsx",
  "status": "success", "rows_extracted": 120, "tokens_in": 15000,
  "tokens_out": 3200, "duration_ms": 8240}
```

---

## Request ID Correlation

Auto-inject a unique request ID into every log line during a request:

```python
"""
Module Name: request_context.py

Description:
    Request ID middleware and log filter for correlation across log lines.
"""

# --------------------------------------------------------------------------
# SECTION: Imports
# --------------------------------------------------------------------------

# Standard library imports
import logging
import uuid
from contextvars import ContextVar

# Third-party imports
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# --------------------------------------------------------------------------
# SECTION: Context Variable
# --------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="no-request-id")

# --------------------------------------------------------------------------
# SECTION: Middleware
# --------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a unique ID to each request, propagates via context var."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Inject request ID into context and response headers."""
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(req_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

# --------------------------------------------------------------------------
# SECTION: Log Filter
# --------------------------------------------------------------------------

class RequestIDFilter(logging.Filter):
    """Injects request_id into every log record automatically."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id field to the log record."""
        record.request_id = request_id_var.get()
        return True

# --------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------
```

**Register in FastAPI:**
```python
app = FastAPI()
app.add_middleware(RequestIDMiddleware)
```

**Update `setup_logger()` formatter to include request ID:**
```python
formatter = logging.Formatter(
    "[%(asctime)s] [%(request_id)s] %(levelname)s in %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.addFilter(RequestIDFilter())
```

**Result — every log line correlates to the same request:**
```
[2026-03-22 10:30:15] [req-a1b2c3] INFO in src.pipeline: Processing file...
[2026-03-22 10:30:16] [req-a1b2c3] INFO in src.agents: Extraction complete
[2026-03-22 10:30:16] [req-a1b2c3] CANONICAL {"event": "pipeline.run", ...}
```

---

## JSON Logging Mode

Toggle structured JSON output via `LOG_FORMAT` environment variable (from `Settings`):

```python
# In setup_logger(), check settings.log_format:

import json as json_lib
import logging

class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize log record to JSON."""
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json_lib.dumps(payload)
```

**In `setup_logger()`:**
```python
import os

log_format = os.getenv("LOG_FORMAT", "text")
formatter = JsonFormatter() if log_format == "json" else logging.Formatter(
    "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

**JSON output:**
```json
{"timestamp": "2026-03-22T10:30:15", "level": "INFO", "module": "src.pipeline", "request_id": "a1b2c3", "message": "Processing file: input.xlsx"}
```

Usage: `LOG_FORMAT=text` (dev) vs `LOG_FORMAT=json` (production/server).

---

## Async Patterns — Extended Examples

**Concurrent API calls with `asyncio.gather()`:**
```python
import asyncio
import aiohttp
from typing import List, Optional

async def fetch_batch(
    urls: List[str],
    timeout: int = 30,
) -> List[Optional[str]]:
    """
    Fetch multiple URLs concurrently.

    Args:
        urls: List of URLs to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        List of response bodies (None on failure).
    """
    connector = aiohttp.TCPConnector(limit=20)
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout_cfg,
    ) as session:
        tasks = [_fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)


async def _fetch_one(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch a single URL, returning None on HTTP error."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None
```

**Wrapping sync code that blocks:**
```python
import asyncio
from functools import partial

async def run_sync_in_thread(func, *args, **kwargs):
    """Run a blocking sync function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

# Usage
result = await run_sync_in_thread(some_sync_library.process, data)
```

**Async context manager:**
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def managed_connection(url: str) -> AsyncGenerator[Connection, None]:
    """Async context manager for database connections."""
    conn = await create_connection(url)
    try:
        yield conn
    finally:
        await conn.close()

# Usage
async with managed_connection(settings.database_url) as conn:
    result = await conn.execute(query)
```

---

## CHANGELOG.md Template

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] — 2026-02-22

### Added
- New feature description (bullet points)
- Include file paths when relevant: `path/to/module.py`

### Changed
- Modifications to existing features; mark breaking changes clearly

### Fixed
- Bug fixes with enough context to understand the change

### Removed
- Previously deprecated features now gone

### Security
- Security updates; include CVE numbers when applicable
```

---

## README.md Template

```markdown
# Project Title

One-sentence description of what the project does.

---

## Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.12 · FastAPI |

---

## Architecture

Brief overview of key flows or design decisions (3–5 sentences or a diagram).

---

## Quick Start

### Prerequisites
- Requirement 1

### Setup
```bash
# Numbered commands — keep it genuinely quick (< 5 steps to running state)
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|

---

## Development

Hot reload notes, migration commands, test commands.

---

## Project Structure

```
project/
├── src/         # Source code
├── tests/       # Tests
└── docs/        # Documentation
```

---

## Docs

- [Architecture](docs/architecture.md)
- [Style Guide](docs/coding_style_guide.md)
```

---

## Jupyter Notebook Conventions

- All notebooks live in `notebooks/` organized by purpose
- File naming: `snake_case`, descriptive, **no dates** in filename — use git history
  - Good: `extraction_analysis.ipynb`, `db_setup.ipynb`
  - Bad: `analysis_2026_02_22.ipynb`
- Standard structure:
  1. Markdown title cell — state the notebook's purpose
  2. Imports cell — same 3-group import pattern as `.py` files
  3. Configuration cell — paths, constants, flags
  4. Processing cells
  5. Results / visualization cells
- Use `setup_logger(__name__)` in notebooks the same as in `.py` files
- Clear large outputs before committing (avoid bloating the repo)

---

## Complete Module Example

A realistic module showing all conventions applied together:

```python
"""
Module Name: order_processor.py

Description:
    Processes incoming customer orders, validates line items against
    inventory, and persists results to the database.
"""

# --------------------------------------------------------------------------
# SECTION: Imports
# --------------------------------------------------------------------------

# Standard library imports
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# Local application imports
from src.logger import setup_logger
from src.database.repositories.order_repository import OrderRepository

# --------------------------------------------------------------------------
# SECTION: Logger Initialization
# --------------------------------------------------------------------------

logger = setup_logger(__name__)

# --------------------------------------------------------------------------
# SECTION: Constants
# --------------------------------------------------------------------------

MAX_RETRIES = 3
DEFAULT_CURRENCY = "USD"

# --------------------------------------------------------------------------
# SECTION: Type Aliases
# --------------------------------------------------------------------------

LineItemType = Dict[str, any]

# --------------------------------------------------------------------------
# SECTION: Schemas
# --------------------------------------------------------------------------

class OrderCreate(BaseModel):
    """Request schema for creating a new order."""

    customer_id: uuid.UUID = Field(description="ID of the customer placing the order.")
    line_items: List[LineItemType] = Field(description="List of items in the order.")
    currency: str = Field(default=DEFAULT_CURRENCY, description="ISO 4217 currency code.")

    model_config = ConfigDict(str_strip_whitespace=True)


class OrderResponse(BaseModel):
    """Response schema — never expose ORM objects directly."""

    order_id: uuid.UUID
    customer_id: uuid.UUID
    total: float
    status: str

    model_config = ConfigDict(from_attributes=True)

# --------------------------------------------------------------------------
# SECTION: Order Processor
# --------------------------------------------------------------------------

class OrderProcessor:
    """
    Validates and processes customer orders.

    Attributes:
        repo (OrderRepository): Database access for order operations.
        config_path (Path): Path to processing configuration.
    """

    # --- Constructor ---
    def __init__(self, repo: OrderRepository, config_path: Path) -> None:
        self.repo = repo
        self.config_path = config_path
        logger.info(f"OrderProcessor initialized with config: {config_path}")

    # --- Public API ---
    def process_order(self, order: OrderCreate) -> OrderResponse:
        """
        Validate and persist a customer order.

        Args:
            order: Validated order creation request.

        Returns:
            Persisted order with generated ID and computed total.

        Raises:
            ValueError: If any line item references invalid inventory.
        """
        self._validate_line_items(order.line_items)
        total = sum(item.get("price", 0) * item.get("qty", 0) for item in order.line_items)

        saved = self._save_with_retry(order, total)
        logger.info(f"Order {saved.order_id} processed — total: {total}")
        return OrderResponse.model_validate(saved)

    # --- Private Helpers ---
    def _validate_line_items(self, items: List[LineItemType]) -> None:
        """Check each line item against inventory."""
        for item in items:
            if not item.get("sku"):
                raise ValueError(f"Line item missing SKU: {item}")

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(max=10))
    def _save_with_retry(self, order: OrderCreate, total: float):
        """Persist order with automatic retry on transient DB errors."""
        return self.repo.create(
            customer_id=order.customer_id,
            line_items=order.line_items,
            total=total,
        )

# --------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------
```

---

## Additional Resources

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [Tenacity — Retrying Library](https://tenacity.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [pydantic-settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [asyncio — Python Docs](https://docs.python.org/3/library/asyncio.html)
- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
- [Semantic Versioning](https://semver.org/)
