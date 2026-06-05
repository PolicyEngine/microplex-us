"""Shared utilities for the PE-US-data checkpoint runner."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("microplex_us.pipelines.pe_us_data_rebuild_checkpoint")


def _root_logger_has_handlers() -> bool:
    return bool(logging.getLogger().handlers)


def _emit_checkpoint_progress(message: str, /, **context: object) -> None:
    details = ", ".join(
        f"{key}={value}"
        for key, value in context.items()
        if value is not None and value != ""
    )
    line = f"{message} [{details}]" if details else message
    LOGGER.info(line)
    if not LOGGER.handlers and not _root_logger_has_handlers():
        print(line, file=sys.stderr, flush=True)


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def _resolve_policyengine_us_runtime_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return None
