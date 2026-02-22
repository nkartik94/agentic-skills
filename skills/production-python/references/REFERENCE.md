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
- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
- [Semantic Versioning](https://semver.org/)
