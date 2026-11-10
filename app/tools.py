"""Tool schemas and implementations.

Design rule for everything in this file: **thin tools, thick model.**
`check_weather` returns numbers, never a verdict like "cold and wet".
`add_item` records a decision, it does not second-guess whether the item makes
sense. All the judgment lives in the model, which is what makes this an agent
rather than a script with a language model bolted on.

The one exception is ordering: `add_item` refuses while no forecast is on
record. That is enforced through the *tool result* rather than `tool_choice`,
so sequencing is still something the model learns and chooses, not something
the harness imposes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from .models import CATEGORIES, Item, Trip, TripState
from .store import Store, find_item, slugify
from .weather import WeatherError, fetch_forecast
from .observability import metrics

logger = logging.getLogger(__name__)

# OpenAI function-calling schemas. Kept small and flat: descriptions are short,
# required args are minimal, and there are only three tools. Small models
# degrade quickly as the tool surface grows.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_trip",
            "description": (
                "Record the trip. Call this first if no trip is on record. "
                "Also call it to correct trip details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Destination city, e.g. Reykjavik",
                    },
                    "trip_type": {
                        "type": "string",
                        "description": (
                            "Short label for the kind of trip, e.g. hiking, "
                            "business, beach, ski, city break"
                        ),
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Departure date as YYYY-MM-DD. Omit for today.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Trip length in days. Omit for 5.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": (
                "Get the real daily forecast for the trip. Call this before "
                "adding any items. Omit city to use the trip on record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Override the city. Omit to use the trip city.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": (
                "Add one item to the packing list. Call once per item. The "
                "reason must cite a specific number or fact from the forecast "
                "or the trip type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The thing to pack, e.g. waterproof shell",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why, citing forecast figures or the trip type, "
                            "e.g. '4 of 5 days show precipitation above 70%'"
                        ),
                    },
                    "category": {
                        "type": "string",
                        "enum": list(CATEGORIES),
                        "description": "Which dimension this item covers",
                    },
                },
                "required": ["item", "reason", "category"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


@dataclass
class ToolOutcome:
    text: str
    is_error: bool = False
    item: Item | None = None


def parse_arguments(raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse tool-call arguments, tolerating small-model sloppiness.

    Returns (args, error_message). Never raises — a malformed call has to come
    back to the model as a readable error so it can retry, because on an 8B
    model malformed JSON is a routine event rather than an exceptional one.
    """
    if raw is None or not str(raw).strip():
        return {}, None

    text = str(raw).strip()

    # Models sometimes wrap arguments in a markdown fence.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Occasionally the arguments arrive double-encoded as a JSON string.
        try:
            once = json.loads(json.dumps(text))
            parsed = json.loads(once)
        except (json.JSONDecodeError, TypeError):
            return None, (
                "Your arguments were not valid JSON. Call the tool again with "
                'a plain JSON object, for example {"city": "Reykjavik"}.'
            )

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return None, (
                "Your arguments were a string, not a JSON object. Retry with "
                "a JSON object."
            )

    if not isinstance(parsed, dict):
        return None, (
            f"Arguments must be a JSON object, got {type(parsed).__name__}. Retry."
        )

    return parsed, None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


# Small models improvise argument names, especially when the surrounding prose
# uses a different word than the schema. Observed on llama3.1:8b: it sent
# `dimension` instead of `category` on ten consecutive calls and never
# recovered from the resulting error. Accepting synonyms costs nothing — the
# model still chooses every *value*, which is where the reasoning lives.
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "add_item": {
        "dimension": "category",
        "dimensions": "category",
        "categories": "category",
        "type": "category",
        "group": "category",
        "section": "category",
        "name": "item",
        "items": "item",
        "thing": "item",
        "object": "item",
        "why": "reason",
        "justification": "reason",
        "rationale": "reason",
        "because": "reason",
        "explanation": "reason",
    },
    "set_trip": {
        "destination": "city",
        "location": "city",
        "place": "city",
        "type": "trip_type",
        "trip": "trip_type",
        "kind": "trip_type",
        "activity": "trip_type",
        "duration": "days",
        "num_days": "days",
        "number_of_days": "days",
        "length": "days",
        "date": "start_date",
        "start": "start_date",
        "departure": "start_date",
        "departure_date": "start_date",
    },
    "check_weather": {
        "destination": "city",
        "location": "city",
        "place": "city",
    },
}


def normalize_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise argument names, preferring correct keys over aliases."""
    aliases = _ARG_ALIASES.get(name, {})
    out: dict[str, Any] = {}
    for key, value in args.items():
        lowered = str(key).strip().lower()
        if lowered not in aliases:
            out[lowered] = value
    for key, value in args.items():
        lowered = str(key).strip().lower()
        canonical = aliases.get(lowered)
        if canonical and not _text(out.get(canonical)):
            out[canonical] = value
    return out


def _wrong_names_hint(args: dict, expected: tuple[str, ...]) -> str:
    got = sorted(str(k) for k in args)
    return (
        f" You sent: {', '.join(got) or '(nothing)'}. "
        f"The argument names must be exactly: {', '.join(expected)}."
    )



# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------


def _set_trip(args: dict, state: TripState, store: Store, timeout: float) -> ToolOutcome:
    city = _text(args.get("city"))
    trip_type = _text(args.get("trip_type")) or (
        state.trip.trip_type if state.trip else "general"
    )

    missing = [n for n, v in (("city", city),) if not v]
    if missing:
        return ToolOutcome(
            f"Missing required argument(s): {', '.join(missing)}."
            + _wrong_names_hint(args, ("city", "trip_type", "start_date", "days")),
            is_error=True,
        )

    start = _text(args.get("start_date")) or date.today().isoformat()
    try:
        date.fromisoformat(start)
    except ValueError:
        return ToolOutcome(
            f"start_date {start!r} is not YYYY-MM-DD. Retry with that format, "
            "or omit it to use today.",
            is_error=True,
        )

    days = max(1, min(_as_int(args.get("days"), 5), 60))

    previous_city = state.trip.city if state.trip else None
    old_trip_id = state.trip_id
    state.trip = Trip(city=city, trip_type=trip_type, start_date=start, days=days)

    if previous_city is None and old_trip_id == "default":
        state.trip_id = slugify(f"{city}-{trip_type}-{start}")

    # A different destination makes the stored forecast meaningless.
    if previous_city and previous_city.lower() != city.lower():
        state.forecast = None

    store.save(state)
    if state.trip_id != old_trip_id:
        store.delete(old_trip_id)

    tail = (
        "A forecast is already on record."
        if state.forecast
        else "No forecast yet — call check_weather next."
    )
    return ToolOutcome(
        f"Trip recorded as '{state.trip_id}': {city}, {trip_type}, "
        f"{days} days from {start}. {tail}"
    )


def _check_weather(
    args: dict, state: TripState, store: Store, timeout: float
) -> ToolOutcome:
    city = _text(args.get("city")) or (state.trip.city if state.trip else "")
    if not city:
        return ToolOutcome(
            "No city given and no trip on record. Call set_trip first, or pass "
            "a city argument.",
            is_error=True,
        )

    start = state.trip.start_date if state.trip else date.today().isoformat()
    days = state.trip.days if state.trip else 5

    try:
        started = time.monotonic()
        forecast = fetch_forecast(city, start, days, timeout=timeout)
    except WeatherError as exc:
        metrics.increment("weather_requests_failed_total")
        logger.warning("weather_request_failed", extra={"event": "weather_request_failed", "error_type": type(exc).__name__, "duration_ms": round((time.monotonic() - started) * 1000)})
        # Deliberately not raised: the model reads this and reports honestly
        # rather than the turn dying on an exception.
        return ToolOutcome(str(exc), is_error=True)

    metrics.increment("weather_requests_succeeded_total")
    logger.info("weather_request_succeeded", extra={"event": "weather_request_succeeded", "duration_ms": round((time.monotonic() - started) * 1000)})

    state.forecast = forecast
    store.save(state)

    where = forecast.city_resolved + (
        f", {forecast.country}" if forecast.country else ""
    )
    lines = [f"Forecast for {where}:"]
    for d in forecast.days:
        lines.append(
            f"  {d.date}: {d.temp_min_c:.0f} to {d.temp_max_c:.0f}C, "
            f"precip {d.precip_prob_pct}% / {d.precip_mm:.1f}mm, "
            f"wind {d.wind_max_kmh:.0f}km/h, {d.condition}"
        )
    lines.append(forecast.summary_line())
    if forecast.note:
        lines.append(f"Note: {forecast.note}")
    lines.append("Now decide what to pack and call add_item once per item.")
    return ToolOutcome("\n".join(lines))


def _add_item(args: dict, state: TripState, store: Store, timeout: float) -> ToolOutcome:
    if state.forecast is None:
        # Sequencing taught through the tool result, not forced by the harness.
        return ToolOutcome(
            "No forecast on record for this trip. Call check_weather first — "
            "items must be justified by real weather data.",
            is_error=True,
        )

    item = _text(args.get("item"))
    reason = _text(args.get("reason"))
    category = _text(args.get("category")).lower().replace(" ", "_")

    expected = ("item", "reason", "category")
    if not item:
        return ToolOutcome(
            "Missing required argument: item."
            + _wrong_names_hint(args, expected),
            is_error=True,
        )
    if not reason:
        return ToolOutcome(
            f"Missing a reason for {item!r}. Retry with a reason citing a "
            "forecast figure or the trip type."
            + _wrong_names_hint(args, expected),
            is_error=True,
        )
    if category not in CATEGORIES:
        detail = (
            f"category {category!r} is not valid."
            if category
            else "The category argument was empty or missing."
        )
        return ToolOutcome(
            f"{detail} Choose one of: {', '.join(CATEGORIES)}."
            + _wrong_names_hint(args, expected),
            is_error=True,
        )

    existing = find_item(state, item)
    if existing:
        return ToolOutcome(
            f'"{existing.item}" is already on the list under '
            f"{existing.category}. Not added again. Move on to something else."
        )

    record = Item(item=item, reason=reason, category=category)
    state.items.append(record)
    store.save(state)

    # Phrased as information, not as a checklist. "Still empty: sleepwear" reads
    # as a quota to clear, and llama3.1:8b duly filed hiking socks under
    # sleepwear to close the gap. Coverage should prompt consideration, never
    # pressure a miscategorised item into an empty slot.
    empty = state.empty_categories()
    tail = (
        "Every category has now been considered."
        if not empty
        else (
            "Not yet considered: "
            + ", ".join(empty)
            + ". Only add to these if this trip genuinely needs them."
        )
    )
    return ToolOutcome(
        f'Added "{item}" [{category}]. {len(state.items)} items packed. {tail}',
        item=record,
    )


_HANDLERS = {
    "set_trip": _set_trip,
    "check_weather": _check_weather,
    "add_item": _add_item,
}


def execute(
    name: str, args: dict, state: TripState, store: Store, timeout: float = 20.0
) -> ToolOutcome:
    handler = _HANDLERS.get(name)
    if handler is None:
        metrics.increment("tool_calls_failed_total")
        metrics.increment(f"tool_calls_{name}_failed_total")
        return ToolOutcome(
            f"No tool named {name!r}. Available tools: "
            f"{', '.join(sorted(TOOL_NAMES))}.",
            is_error=True,
        )
    try:
        outcome = handler(normalize_args(name, args or {}), state, store, timeout)
        outcome_name = "failed" if outcome.is_error else "succeeded"
        metrics.increment(f"tool_calls_{outcome_name}_total")
        metrics.increment(f"tool_calls_{name}_{outcome_name}_total")
        return outcome
    except Exception as exc:  # last-resort net so one bad call cannot end the turn
        metrics.increment("tool_calls_failed_total")
        metrics.increment(f"tool_calls_{name}_failed_total")
        logger.exception("tool_execution_failed", extra={"event": "tool_execution_failed", "tool": name})
        return ToolOutcome(
            f"{name} failed unexpectedly: {type(exc).__name__}: {exc}", is_error=True
        )
