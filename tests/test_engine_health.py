from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from urllib.request import urlopen

import openai

from app.engine import PackingAgent
from app.health import ObservabilityServer
from app.observability import JsonFormatter, configure_logging, metrics


def response(content="", calls=()):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=list(calls)))])


class FakeCompletions:
    def __init__(self, responses): self.responses = iter(responses); self.calls = 0
    def create(self, **kwargs): self.calls += 1; return next(self.responses)


class FakeClient:
    def __init__(self, responses): self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_agent_stops_at_max_iterations(settings, store):
    client = FakeClient([response(calls=[tool_call("set_trip", "{")]) for _ in range(3)])
    events = list(PackingAgent(settings, store, client).send("x", "plan"))
    assert any(e.type == "warning" and "Stopped after" in (e.text or "") for e in events)


def test_agent_detects_duplicate_call_and_failed_item_write(settings, store):
    client = FakeClient([
        response(calls=[tool_call("add_item", '{"item":"coat","reason":"rain","category":"outerwear"}'), tool_call("add_item", '{"item":"coat","reason":"rain","category":"outerwear"}', "call-2")]),
        response("I added a coat."),
        response("I added a coat."),
    ])
    events = list(PackingAgent(settings, store, client).send("x", "plan"))
    assert any("already made this exact call" in (e.result or "") for e in events)
    assert any(e.type == "warning" and "all 2 add_item" in (e.text or "") for e in events)


def test_agent_returns_model_connection_error(settings, store):
    class Failing:
        def create(self, **kwargs):
            raise openai.APIConnectionError(request=__import__("httpx").Request("GET", "http://model.invalid"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=Failing()))
    events = list(PackingAgent(settings, store, client).send("x", "plan"))
    assert any(e.type == "error" and "Cannot reach the model" in (e.text or "") for e in events)


def test_health_endpoints(settings):
    server = ObservabilityServer("127.0.0.1", 0, settings.data_dir)
    port = server.server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert urlopen(f"http://127.0.0.1:{port}/healthz").read() == b"ok\n"
        assert urlopen(f"http://127.0.0.1:{port}/readyz").status == 200
        metrics.increment("turns_total")
        assert b"packing_agent_turns_total" in urlopen(f"http://127.0.0.1:{port}/metrics").read()
    finally:
        server.close()


def test_json_logging_does_not_include_unprovided_secret():
    import logging
    configure_logging("INFO", "json")
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)
    record = logging.LogRecord("app", logging.INFO, "", 0, "started", (), None)
    encoded = JsonFormatter().format(record)
    assert json.loads(encoded)["message"] == "started"
    assert "test-key" not in encoded
