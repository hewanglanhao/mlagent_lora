from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass
class BackgroundTask(Generic[T]):
    name: str
    metadata: dict[str, Any]
    started_ts: float = field(default_factory=time.time)
    finished_ts: float | None = None
    result: T | None = None
    error: str | None = None
    _done: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def done(self) -> bool:
        return self._done.is_set()

    def duration_sec(self) -> float:
        end = self.finished_ts if self.finished_ts is not None else time.time()
        return end - self.started_ts


def start_background_task(
    name: str,
    func: Callable[[], T],
    metadata: dict[str, Any] | None = None,
) -> BackgroundTask[T]:
    task: BackgroundTask[T] = BackgroundTask(name=name, metadata=metadata or {})

    def run() -> None:
        try:
            task.result = func()
        except Exception as exc:  # pragma: no cover - defensive boundary for worker threads
            task.error = f"{type(exc).__name__}: {exc}"
        finally:
            task.finished_ts = time.time()
            task._done.set()

    task.thread = threading.Thread(target=run, name=f"mlagent-{name}", daemon=True)
    task.thread.start()
    return task
