from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .llm_adapter import LLMClient, LLMResponse
from .memory import atomic_write_text


class LLMCallLogger:
    """Records every LLM request/response attempt for one run."""

    def __init__(self, root: Path, run_id: str, logger: logging.Logger) -> None:
        self.root = root
        self.run_id = run_id
        self.logger = logger
        self.call_dir = root / "experiments" / run_id / "llm_calls"
        self.call_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.call_dir / "manifest.jsonl"
        self.preview_chars = int(os.getenv("LLM_LOG_PREVIEW_CHARS", "4000"))
        self.include_full = os.getenv("LLM_LOG_FULL_TEXT", "0") == "1"
        self._counter = 0
        self._paths: dict[str, Path] = {}

    def complete(
        self,
        llm: LLMClient,
        agent_name: str,
        system: str,
        user: str,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse | None:
        self._counter += 1
        call_id = f"{self._counter:04d}_{self._safe_name(agent_name)}_{int(time.time() * 1000)}"
        path = self.call_dir / f"{call_id}.json"
        self._paths[call_id] = path
        started = time.time()
        record: dict[str, Any] = {
            "call_id": call_id,
            "run_id": self.run_id,
            "agent": agent_name,
            "status": "started",
            "started_ts": started,
            "model": llm.model,
            "llm_status": llm.status,
            "metadata": metadata or {},
            "request": {
                "system_chars": len(system),
                "user_chars": len(user),
                "system_sha256": self._sha256(system),
                "user_sha256": self._sha256(user),
                "system_preview": self._preview(system),
                "user_preview": self._preview(user),
            },
        }
        if self.include_full:
            record["request"]["system_full"] = system
            record["request"]["user_full"] = user
        self._write_record(path, record)
        self._append_manifest(record)
        self.logger.info("llm_call_start id=%s agent=%s model=%s", call_id, agent_name, llm.model)

        response = llm.complete(system=system, user=user)
        finished = time.time()
        record["finished_ts"] = finished
        record["duration_sec"] = finished - started

        if response is None:
            record["status"] = "error_or_empty"
            record["error"] = llm.last_request_error or "LLM returned no response."
            self._write_record(path, record)
            self._append_manifest(self._manifest_event(record))
            self.logger.warning(
                "llm_call_empty id=%s agent=%s duration=%.2fs error=%s",
                call_id,
                agent_name,
                record["duration_sec"],
                record["error"],
            )
            return None

        logged_response = LLMResponse(content=response.content, model=response.model, call_id=call_id)
        record["status"] = "success"
        record["response"] = {
            "model": response.model,
            "content_chars": len(response.content),
            "content_sha256": self._sha256(response.content),
            "content_preview": self._preview(response.content),
        }
        if self.include_full:
            record["response"]["content_full"] = response.content
        self._write_record(path, record)
        self._append_manifest(self._manifest_event(record))
        self.logger.info(
            "llm_call_success id=%s agent=%s duration=%.2fs response_chars=%s",
            call_id,
            agent_name,
            record["duration_sec"],
            len(response.content),
        )
        return logged_response

    def record_parse(self, call_id: str | None, parser: str, ok: bool, detail: str = "") -> None:
        if not call_id:
            return
        path = self._paths.get(call_id)
        if path is None or not path.exists():
            return
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        parse_record = {
            "parser": parser,
            "ok": ok,
            "detail": detail,
            "ts": time.time(),
        }
        record.setdefault("parse_results", []).append(parse_record)
        self._write_record(path, record)
        self._append_manifest({"call_id": call_id, "event": "parse", **parse_record})
        if ok:
            self.logger.info("llm_call_parse_ok id=%s parser=%s", call_id, parser)
        else:
            self.logger.warning("llm_call_parse_failed id=%s parser=%s detail=%s", call_id, parser, detail)

    def _write_record(self, path: Path, record: dict[str, Any]) -> None:
        atomic_write_text(path, json.dumps(record, indent=2, sort_keys=True) + "\n")

    def _append_manifest(self, event: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(event, sort_keys=True) + "\n")

    def _manifest_event(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "call_id": record["call_id"],
            "run_id": record["run_id"],
            "agent": record["agent"],
            "status": record["status"],
            "duration_sec": record.get("duration_sec"),
            "error": record.get("error"),
            "response_chars": record.get("response", {}).get("content_chars"),
        }

    def _preview(self, text: str) -> str:
        if len(text) <= self.preview_chars:
            return text
        return text[: self.preview_chars] + "\n...<trimmed>..."

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _safe_name(self, name: str) -> str:
        return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)
