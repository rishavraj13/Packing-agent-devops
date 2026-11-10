from __future__ import annotations

import httpx
import pytest

from app.models import TripState
from app.tools import execute, normalize_args, parse_arguments
from app.weather import WeatherError, fetch_forecast


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


class FakeClient:
    def __init__(self, responses): self.responses = iter(responses)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def get(self, *args, **kwargs): return next(self.responses)


def test_weather_success(monkeypatch):
    monkeypatch.setattr("app.weather.httpx.Client", lambda **_: FakeClient([
        FakeResponse({"results": [{"name": "Paris", "country": "France", "latitude": 1, "longitude": 2}]}),
        FakeResponse({"timezone": "UTC", "daily": {"time": ["2026-09-01"], "temperature_2m_min": [10], "temperature_2m_max": [20], "precipitation_probability_max": [1], "precipitation_sum": [0], "wind_speed_10m_max": [5], "weather_code": [0]}}),
    ]))
    forecast = fetch_forecast("Paris", "2026-09-01", 1)
    assert forecast.days[0].temp_max_c == 20


@pytest.mark.parametrize("error", [httpx.TimeoutException("late"), httpx.ConnectError("offline")])
def test_weather_network_errors_are_domain_errors(monkeypatch, error):
    class FailingClient(FakeClient):
        def get(self, *args, **kwargs): raise error
    monkeypatch.setattr("app.weather.httpx.Client", lambda **_: FailingClient([]))
    with pytest.raises(WeatherError):
        fetch_forecast("Paris", "2026-09-01", 1)


def test_weather_rejects_no_geocode_and_malformed_daily(monkeypatch):
    monkeypatch.setattr("app.weather.httpx.Client", lambda **_: FakeClient([FakeResponse({})]))
    with pytest.raises(WeatherError): fetch_forecast("Nowhere", "2026-09-01", 1)


def test_tool_validation_and_forecast_gate(store):
    assert normalize_args("add_item", {"name": "coat", "why": "rain", "dimension": "outerwear"})["item"] == "coat"
    assert parse_arguments("{")[0] is None
    outcome = execute("add_item", {"item": "coat", "reason": "rain", "category": "outerwear"}, TripState(trip_id="x"), store)
    assert outcome.is_error and "check_weather first" in outcome.text
