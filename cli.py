"""Terminal REPL for the packing-list agent."""

from __future__ import annotations

import argparse
import sys

from app.config import load_settings
from app.engine import PackingAgent
from app.models import CATEGORIES, Event, TripState
from app.store import Store

HELP = """
Commands
  <anything>   talk to the agent, e.g. "5 days in Reykjavik, hiking"
  /check       forgot something? audit the list against trip + forecast
  /list        show the packing list
  /state       show everything on record for this trip
  /trips       list saved trips
  /use <name>  switch to another trip (creates it if new)
  /reset       delete this trip's saved data
  /help        this
  /quit        exit
"""

def render(event: Event) -> None:
    if event.type == "assistant_message":
        print(f"\nagent> {event.text}")
    elif event.type == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in (event.args or {}).items())
        print(f"  [call] {event.tool}({args})")
    elif event.type == "tool_result":
        mark = "!!" if event.is_error else "->"
        first, *rest = (event.result or "").splitlines() or [""]
        print(f"  {mark} {first}")
        for line in rest:
            print(f"     {line}")
    elif event.type == "item_added":
        item = event.item
        if item:
            print(f"  [+] {item.item}  [{item.category}]")
    elif event.type == "warning":
        print(f"  (!) {event.text}")
    elif event.type == "error":
        print(f"\n[error] {event.text}")


def print_list(state: TripState) -> None:
    if not state.items:
        print("\nNothing packed yet.")
        return
    print(f"\nPacking list — {len(state.items)} items")
    for category in CATEGORIES:
        group = [i for i in state.items if i.category == category]
        if not group:
            continue
        print(f"\n  {category}")
        for item in group:
            print(f"    - {item.item}")
            print(f"        {item.reason}")
    empty = state.empty_categories()
    if empty:
        print("\n  nothing packed for: " + ", ".join(empty))


def print_state(state: TripState) -> None:
    from app.store import render_state_block

    print()
    print(render_state_block(state))


def main() -> int:
    # Windows consoles default to cp1252 and mangle the dashes and degree signs
    # that appear in forecasts and tool messages.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Smart packing list agent")
    parser.add_argument(
        "--trip", default="default", help="trip id (one JSON file per trip)"
    )
    parser.add_argument(
        "message", nargs="*", help="optional first message; omit for a bare prompt"
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    store = Store(settings.data_dir)
    agent = PackingAgent(settings, store=store)
    trip_id = args.trip

    print(f"Packing agent — model {settings.model} at {settings.base_url}")
    print(f"Trip '{trip_id}'. /help for commands.")

    pending = " ".join(args.message).strip() or None

    while True:
        if pending is None:
            try:
                raw = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
        else:
            raw, pending = pending, None
            print(f"\nyou> {raw}")

        if not raw:
            continue

        if raw.startswith("/"):
            command, _, rest = raw.partition(" ")
            command, rest = command.lower(), rest.strip()

            if command in ("/quit", "/exit", "/q"):
                return 0
            if command == "/help":
                print(HELP)
                continue
            if command == "/list":
                print_list(store.load(trip_id))
                continue
            if command == "/state":
                print_state(store.load(trip_id))
                continue
            if command == "/trips":
                trips = store.list_trips()
                print("\n" + ("\n".join(f"  {t}" for t in trips) if trips else "  none"))
                continue
            if command == "/use":
                if not rest:
                    print("  usage: /use <name>")
                    continue
                trip_id = rest
                print(f"  switched to trip '{trip_id}'")
                continue
            if command == "/reset":
                print(
                    f"  deleted '{trip_id}'"
                    if store.delete(trip_id)
                    else f"  nothing saved for '{trip_id}'"
                )
                continue
            if command == "/check":
                for event in agent.review(trip_id):
                    render(event)
                continue

            print(f"  unknown command {command}. /help for the list.")
            continue

        for event in agent.send(trip_id, raw):
            render(event)


if __name__ == "__main__":
    raise SystemExit(main())
