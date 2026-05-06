from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, path)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=dst.parent) as tmp:
        with src.open("rb") as in_file:
            shutil.copyfileobj(in_file, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, dst)


class ExperimentMemory:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = run_id
        self.base_dir = root / "experiments"
        self.experiments_dir = self.base_dir / run_id
        self.trace_path = self.experiments_dir / "agent_trace.jsonl"
        self.leaderboard_path = self.experiments_dir / "leaderboard.json"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.base_dir / "LATEST", run_id + "\n")
        self.write_json(self.experiments_dir / "run_metadata.json", {"run_id": run_id, "root": str(root)})

    def experiment_dir(self, experiment_id: int) -> Path:
        path = self.experiments_dir / f"exp_{experiment_id:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: Path, data: Any) -> None:
        atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    def append_trace(self, event: dict[str, Any]) -> None:
        record = {"ts": time.time(), "run_id": self.run_id, **event}
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")

    def update_leaderboard(self, records: list[dict[str, Any]]) -> None:
        ordered = sorted(
            records,
            key=lambda item: item.get("aggregate_speedup") or float("-inf"),
            reverse=True,
        )
        self.write_json(self.leaderboard_path, ordered)
