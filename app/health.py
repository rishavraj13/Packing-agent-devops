"""Optional lightweight HTTP endpoints for container orchestration."""

from __future__ import annotations

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .observability import metrics

logger = logging.getLogger(__name__)


def data_dir_ready(data_dir: Path) -> bool:
    """Check that state storage can be used without contacting external services."""
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".readiness-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def make_handler(data_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            logger.debug("health_request", extra={"event": "health_request", "path": self.path})

        def _send(self, status: HTTPStatus, body: str, content_type: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._send(HTTPStatus.OK, "ok\n", "text/plain; charset=utf-8")
            elif self.path == "/readyz":
                ready = data_dir_ready(data_dir)
                self._send(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                           "ready\n" if ready else "data directory unavailable\n",
                           "text/plain; charset=utf-8")
            elif self.path == "/metrics":
                self._send(HTTPStatus.OK, metrics.prometheus(), "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._send(HTTPStatus.NOT_FOUND, "not found\n", "text/plain; charset=utf-8")
    return Handler


class ObservabilityServer:
    def __init__(self, host: str, port: int, data_dir: Path) -> None:
        self.server = ThreadingHTTPServer((host, port), make_handler(data_dir))
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.server.serve_forever, name="observability", daemon=True)
        self._thread.start()
        logger.info("observability_server_started", extra={"event": "observability_server_started"})

    def serve_forever(self) -> None:
        logger.info("observability_server_started", extra={"event": "observability_server_started"})
        self.server.serve_forever()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("observability_server_stopped", extra={"event": "observability_server_stopped"})
