"""Data model for persisted trips and CLI events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# The fixed dimensions of a packing list. These are deliberately *dimensions*,
# not items — the model decides what (if anything) each one needs for a given
# trip and forecast. A fixed list of items would make this a script; a fixed
# list of dimensions just stops it forgetting that shoes exist.
CATEGORIES: tuple[str, ...] = (
    "outerwear",
    "layers",
    "footwear",
    "sleepwear",
    "activity_gear",
    "toiletries",
    "medication",
    "tech",
    "documents",
    "bags",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DayForecast(BaseModel):
    date: str
    temp_min_c: float
    temp_max_c: float
    precip_prob_pct: int
    precip_mm: float
    wind_max_kmh: float
    condition: str


class Forecast(BaseModel):
    """Raw-ish weather facts. Deliberately carries no verdict.

    No `is_cold` field, no `needs_raincoat` field. The moment this object
    starts holding conclusions, the tool is doing the reasoning and the
    model is only formatting it.
    """

    city_query: str
    city_resolved: str
    country: str = ""
    latitude: float
    longitude: float
    timezone_name: str = ""
    fetched_at: str = Field(default_factory=now_iso)
    days: list[DayForecast] = Field(default_factory=list)
    note: str = ""

    def summary_line(self) -> str:
        """Aggregate numbers only — min/max/counts, never an interpretation."""
        if not self.days:
            return "No daily data."
        lo = min(d.temp_min_c for d in self.days)
        hi = max(d.temp_max_c for d in self.days)
        wet = sum(1 for d in self.days if d.precip_prob_pct >= 50)
        wind = max(d.wind_max_kmh for d in self.days)
        return (
            f"Range {lo:.0f} to {hi:.0f}C. "
            f"Days with precipitation probability >=50%: {wet}/{len(self.days)}. "
            f"Peak wind {wind:.0f} km/h."
        )


class Item(BaseModel):
    item: str
    reason: str
    category: str
    added_at: str = Field(default_factory=now_iso)


class Trip(BaseModel):
    city: str
    trip_type: str
    start_date: str
    days: int


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    at: str = Field(default_factory=now_iso)


class TripState(BaseModel):
    """Everything persisted for one trip. This *is* the agent's memory."""

    trip_id: str
    trip: Trip | None = None
    forecast: Forecast | None = None
    items: list[Item] = Field(default_factory=list)
    chat: list[ChatTurn] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def empty_categories(self) -> list[str]:
        """Dimensions with nothing packed yet.

        Set arithmetic over CATEGORIES — this is what makes the
        "forgot something?" check deterministic about *what to examine*
        while leaving *what to pack* to the model.
        """
        used = {i.category for i in self.items}
        return [c for c in CATEGORIES if c not in used]


# --------------------------------------------------------------------------
# Events. The engine yields these instead of printing so the CLI can render
# progress as it happens.
# --------------------------------------------------------------------------

EventType = Literal[
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "item_added",
    "state",
    "warning",
    "error",
    "done",
]


class Event(BaseModel):
    type: EventType
    text: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    is_error: bool = False
    item: Item | None = None
    state: TripState | None = None
    iteration: int | None = None
