from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import DayForecast, Item, Trip, TripState
from app.store import Store, render_state_block


def test_models_serialize_and_render_state():
    state = TripState(trip_id="demo", trip=Trip(city="Paris", trip_type="city", start_date="2026-09-01", days=3))
    state.items.append(Item(item="shoes", reason="city trip", category="footwear"))
    restored = TripState.model_validate_json(state.model_dump_json())
    assert restored.trip and restored.trip.city == "Paris"
    assert "shoes [footwear]" in render_state_block(restored)
    with pytest.raises(ValidationError):
        DayForecast(date="2026-09-01", temp_min_c=1, temp_max_c=2, precip_prob_pct="bad", precip_mm=0, wind_max_kmh=1, condition="clear")


def test_store_persists_lists_deletes_and_recovers_corruption(tmp_path):
    store = Store(tmp_path)
    assert store.load("missing").trip_id == "missing"
    state = TripState(trip_id="my trip")
    store.save(state)
    assert Store(tmp_path).load("my trip").trip_id == "my trip"
    assert store.list_trips() == ["my-trip"]
    store.path_for("broken").write_text("not json", encoding="utf-8")
    assert store.load("broken").trip_id == "broken"
    assert store.path_for("broken").with_suffix(".json.corrupt").exists()
    assert store.delete("my trip") is True
    assert store.delete("my trip") is False
