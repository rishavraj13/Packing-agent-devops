"""JSON persistence, one file per trip.

Also owns `render_state_block`, which turns stored state back into prompt
text. That function is the reason memory works across runs: the store is
re-read and re-injected on every turn, so nothing important depends on the
chat transcript surviving.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Item, TripState, now_iso

_SAFE = re.compile(r"[^a-z0-9_-]+")


def slugify(text: str) -> str:
    slug = _SAFE.sub("-", text.strip().lower()).strip("-")
    return slug or "trip"


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, trip_id: str) -> Path:
        return self.data_dir / f"{slugify(trip_id)}.json"

    def load(self, trip_id: str) -> TripState:
        path = self.path_for(trip_id)
        if not path.exists():
            return TripState(trip_id=trip_id)
        try:
            return TripState.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt file should not brick the session. Move it aside so
            # the user can inspect it, and start clean.
            path.replace(path.with_suffix(".json.corrupt"))
            return TripState(trip_id=trip_id)

    def save(self, state: TripState) -> None:
        state.updated_at = now_iso()
        path = self.path_for(state.trip_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            state.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
        tmp.replace(path)

    def list_trips(self) -> list[str]:
        return sorted(p.stem for p in self.data_dir.glob("*.json"))

    def delete(self, trip_id: str) -> bool:
        path = self.path_for(trip_id)
        if path.exists():
            path.unlink()
            return True
        return False


def find_item(state: TripState, name: str) -> Item | None:
    needle = name.strip().lower()
    return next((i for i in state.items if i.item.strip().lower() == needle), None)


def render_state_block(state: TripState) -> str:
    """Stored state as compact prompt text.

    Kept terse on purpose: at Ollama's default 4096-token window this block is
    replayed every turn, so every line has to earn its place.
    """
    lines: list[str] = []

    if state.trip is None:
        lines.append("TRIP: not on record yet. Call set_trip before anything else.")
    else:
        t = state.trip
        lines.append(
            f"TRIP: {t.city} | type={t.trip_type} | "
            f"start={t.start_date} | {t.days} days"
        )

    if state.forecast is None:
        lines.append("FORECAST: none on record. Call check_weather before adding items.")
    else:
        f = state.forecast
        where = f.city_resolved + (f", {f.country}" if f.country else "")
        lines.append(f"FORECAST for {where} (fetched {f.fetched_at}):")
        for d in f.days:
            lines.append(
                f"  {d.date}: {d.temp_min_c:.0f} to {d.temp_max_c:.0f}C, "
                f"precip {d.precip_prob_pct}% / {d.precip_mm:.1f}mm, "
                f"wind {d.wind_max_kmh:.0f}km/h, {d.condition}"
            )
        lines.append(f"  {f.summary_line()}")
        if f.note:
            lines.append(f"  Note: {f.note}")

    if not state.items:
        lines.append("PACKED: nothing yet.")
    else:
        lines.append(f"PACKED ({len(state.items)}):")
        for i in state.items:
            lines.append(f"  - {i.item} [{i.category}] because {i.reason}")

    empty = state.empty_categories()
    if empty:
        lines.append("CATEGORIES WITH NOTHING PACKED YET: " + ", ".join(empty))
    else:
        lines.append("CATEGORIES WITH NOTHING PACKED YET: none.")

    return "\n".join(lines)
