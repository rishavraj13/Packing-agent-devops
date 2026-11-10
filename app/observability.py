"""Structured logging and low-cardinality process metrics."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Small dependency-free JSON formatter suitable for stdout collection."""

    _STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", output_format: str = "json") -> None:
    """Configure application logging without exposing configuration secrets."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if output_format == "json" else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class Metrics:
    def __init__(self) -> None:
        self.started_at = time.time()
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += amount

    def value(self, name: str) -> int:
        with self._lock:
            return self._counts[name]

    def prometheus(self) -> str:
        with self._lock:
            counts = dict(self._counts)
        lines = [
            "# HELP packing_agent_process_start_time_seconds Unix time when the process started.",
            "# TYPE packing_agent_process_start_time_seconds gauge",
            f"packing_agent_process_start_time_seconds {self.started_at:.3f}",
        ]
        for name in sorted(counts):
            metric = "packing_agent_" + name.replace(".", "_").replace("-", "_")
            lines.extend((f"# TYPE {metric} counter", f"{metric} {counts[name]}"))
        return "\n".join(lines) + "\n"


metrics = Metrics()
