from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _write_latest(root: Path, run_id: str) -> None:
    latest_path = root / "logs" / "LATEST"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(run_id + "\n", encoding="utf-8")


def setup_runtime_logger(root: Path, run_id: str) -> logging.Logger:
    """Configure the human-readable runtime log for the optimizer."""
    logger = logging.getLogger(f"mlagent_lora.{run_id}")
    if logger.handlers:
        return logger

    log_dir = root / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"
    _write_latest(root, run_id)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(os.getenv("LOG_MAX_BYTES", str(8 * 1024 * 1024))),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", "3")),
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if os.getenv("LOG_TO_STDOUT", "1") != "0":
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info("runtime logger initialized at %s", log_path)
    logger.info("run_id=%s", run_id)
    return logger
