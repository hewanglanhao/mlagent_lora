from __future__ import annotations

import os
from datetime import datetime, timezone


def make_run_id() -> str:
    """Create a filesystem-friendly id for one optimizer invocation."""
    configured = os.getenv("RUN_ID", "").strip()
    if configured:
        return _sanitize(configured)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_pid{os.getpid()}"


def _sanitize(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._")
    return result or "run"

