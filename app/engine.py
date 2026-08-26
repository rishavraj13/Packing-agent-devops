"""The agentic loop.

It yields `Event` objects instead of printing so the CLI can render progress
as each model and tool call completes. It holds no session state: everything
lives in the store, keyed by trip_id.

The loop is hand-written because the OpenAI library has no agentic-loop
helper, and because a small local model needs guardrails at every step where
it can misbehave: malformed argument JSON, tool calls emitted as prose,
repeated identical calls, and runaway iteration. Each of those is handled by
feeding a readable correction back as a tool result rather than raising.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from typing import Any

import openai
from openai import OpenAI

from .config import Settings
from .models import ChatTurn, Event, TripState
from .store import Store, render_state_block
from .tools import TOOL_SCHEMAS, ToolOutcome, execute, parse_arguments

SYSTEM_PROMPT = """You are a packing-list agent. You decide what to pack by \
reasoning about real forecast data, never from a generic list.

How to work:
1. First classify the user's request. Informational questions about the current \
trip, the list, or the forecast are read-only: answer from CURRENT STATE and stop. \
Never create, update, or pack a trip for an informational question.
2. If no trip is on record and the user asks to plan or pack, call set_trip.
3. Call check_weather before adding any items. Never guess or assume the weather.
4. Then call add_item once per item. Every call needs a reason citing a specific \
number from the forecast or a fact about the trip type.
5. Cover these categories, deciding for each whether this trip needs anything: \
outerwear, layers, footwear, sleepwear, activity_gear, toiletries, medication, \
tech, documents, bags. The category list is fixed; what goes in each one is your \
decision, and a category can legitimately need nothing.
6. Scale quantities to the number of days.
7. Do not describe items in prose before calling add_item. The tool call is the \
decision, and an item you only mention in text does not exist.
8. When the categories are handled, write one short paragraph summarising the \
list, then stop.

Every add_item call uses exactly these three argument names:
  item, reason, category

The CURRENT STATE message shows what is already recorded. Never re-add \
something already packed."""

REVIEW_PROMPT = """Review the packed list against the trip and forecast in \
CURRENT STATE.

For each category listed as having nothing packed, decide whether this trip \
genuinely needs something there — some legitimately do not, and padding the \
list is worse than leaving a gap. Add only what is genuinely missing, using \
add_item with a reason citing the forecast or trip type.

Then say briefly what you added and what you deliberately left out and why. If \
nothing is missing, say so and stop."""

# Shapes a small model produces when it means to call a tool but emits text.
# llama3.1 in particular falls back to <function=name>{...}</function> or a
# bare JSON blob when its tool template does not fire.
_TEXT_TOOL_CALL = re.compile(
    r"""(
        <\s*function [^>]* > |
        <\|python_tag\|> |
        "\s*name\s*"\s*:\s*"(?:set_trip|check_weather|add_item)" |
        \b(?:set_trip|check_weather|add_item)\s*\(\s*(?:\{|["'\w])
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_NUDGE = (
    "That looked like a tool call written as text. Tool calls must go through "
    "the tool-calling interface, not the message body. Try again."
)

# The dangerous failure is not an errored tool call — it is the model summarising
# items it never successfully saved, which reads like success and is not.
_ADD_RECOVERY = (
    "None of your add_item calls succeeded, so the packing list is still empty "
    "and the items you just listed were NOT saved. Call add_item again for each "
    'item, using exactly these argument names: {"item": "...", "reason": "...", '
    '"category": "..."}. Do not write the list as prose.'
)


def build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.request_timeout,
        max_retries=1,
    )


class _LockPool:
    """One reentrant lock per trip_id, created on demand.

    A turn mutates the shared TripState across many tool rounds, each of which
    saves to disk. Two turns on the same trip would both load the same base
    state and the second save would silently discard the first's items. So a
    turn holds its trip's lock for its whole duration.

    Reentrant because _run may re-enter through review()/send() on the same
    trip from the same thread. Coarse and in-process only: it guards against
    overlapping turns in one process, not across processes.
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def get(self, trip_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(trip_id, threading.RLock())


class PackingAgent:
    def __init__(
        self,
        settings: Settings,
        store: Store | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or Store(settings.data_dir)
        self.client = client or build_client(settings)
        self._locks = _LockPool()

    # -- public API ---------------------------------------------------------

    def state(self, trip_id: str) -> TripState:
        return self.store.load(trip_id)

    def send(self, trip_id: str, user_message: str) -> Iterator[Event]:
        """Handle one user message. Yields events as work happens."""
        with self._locks.get(trip_id):
            state = self.store.load(trip_id)
            yield Event(type="user_message", text=user_message)
            yield from self._run(state, user_message, persist_user_turn=True)

    def review(self, trip_id: str) -> Iterator[Event]:
        """The "forgot something?" pass.

        Deterministic about *which categories to examine* (the state block
        lists the empty ones, computed in code) and model-driven about *what
        belongs in them* — which is the split that keeps this agentic.
        """
        with self._locks.get(trip_id):
            state = self.store.load(trip_id)
            if state.trip is None:
                yield Event(
                    type="error",
                    text="No trip on record yet — describe your trip first.",
                )
                return
            if state.forecast is None:
                yield Event(
                    type="error",
                    text="No forecast on record yet. Ask for a packing list first.",
                )
                return
            yield from self._run(state, REVIEW_PROMPT, persist_user_turn=False)

    # -- loop --------------------------------------------------------------

    def _build_messages(self, state: TripState, user_message: str) -> list[dict]:
        """System prompt, then state, then a short tail of chat, then the ask.

        The state block is rebuilt from disk every turn, which is what lets
        this work inside a 4096-token window: correctness depends on the store,
        not on the transcript surviving.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "CURRENT STATE\n" + render_state_block(state)},
        ]
        tail = state.chat[-self.settings.history_turns :] if state.chat else []
        messages.extend({"role": t.role, "content": t.content} for t in tail)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _run(
        self, state: TripState, user_message: str, persist_user_turn: bool
    ) -> Iterator[Event]:
        messages = self._build_messages(state, user_message)

        if persist_user_turn:
            state.chat.append(ChatTurn(role="user", content=user_message))
            self.store.save(state)

        final_text: list[str] = []
        # (name, arguments) already executed this turn — small models repeat
        # themselves, and re-running a network call for an identical request is
        # pure waste. Stores the whole outcome, not just its text, so a replay
        # keeps the original error status.
        executed: dict[tuple[str, str], ToolOutcome] = {}
        nudges = 0
        adds_ok = 0
        adds_failed = 0
        recoveries = 0

        for iteration in range(1, self.settings.max_iterations + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                )
            except openai.APIConnectionError as exc:
                yield Event(
                    type="error",
                    text=(
                        f"Cannot reach the model at {self.settings.base_url} "
                        f"({exc.__class__.__name__}). Is the server running?"
                    ),
                )
                return
            except openai.APIStatusError as exc:
                yield Event(
                    type="error",
                    text=f"Model returned HTTP {exc.status_code}: {exc.message}",
                )
                return
            except openai.OpenAIError as exc:
                yield Event(
                    type="error", text=f"{exc.__class__.__name__}: {exc}"
                )
                return

            choice = response.choices[0]
            message = choice.message
            content = (message.content or "").strip()
            tool_calls = list(message.tool_calls or [])

            if content and not tool_calls:
                # Only surface prose as an answer when it is not a mangled
                # tool call.
                if _TEXT_TOOL_CALL.search(content) and nudges < 2:
                    nudges += 1
                    yield Event(
                        type="warning",
                        text="Model wrote a tool call as text; asking it to retry.",
                        iteration=iteration,
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "system", "content": _NUDGE})
                    continue

                # Refuse to accept a summary of items that were never saved.
                if adds_failed and not adds_ok and recoveries < 2:
                    recoveries += 1
                    yield Event(
                        type="warning",
                        text=(
                            f"Model wrapped up but all {adds_failed} add_item "
                            "call(s) failed — nothing was saved. Asking it to "
                            "redo them."
                        ),
                        iteration=iteration,
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "system", "content": _ADD_RECOVERY})
                    continue

                # Recovery attempts exhausted. Never hand back an unqualified
                # summary of items that were not saved — a misleading success
                # only assistant_message would show a clean lie. The warning
                # events alone are not enough, since a consumer may drop them.
                if adds_failed and not adds_ok:
                    content = (
                        content
                        + "\n\n[Nothing above was actually saved: all "
                        f"{adds_failed} attempts to record an item failed. "
                        "The packing list is still empty.]"
                    )

                yield Event(
                    type="assistant_message", text=content, iteration=iteration
                )
                final_text.append(content)
                break

            if not tool_calls:
                # Nothing said, nothing called.
                if nudges < 2:
                    nudges += 1
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "You returned an empty response. Either call a "
                                "tool or answer the user."
                            ),
                        }
                    )
                    continue
                yield Event(
                    type="warning",
                    text="Model returned nothing twice; ending the turn.",
                    iteration=iteration,
                )
                break

            if content:
                yield Event(
                    type="assistant_message", text=content, iteration=iteration
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                name = call.function.name or "<unnamed>"
                args, parse_error = parse_arguments(call.function.arguments)

                yield Event(
                    type="tool_call",
                    tool=name,
                    args=args if args is not None else {},
                    iteration=iteration,
                )

                if parse_error is not None:
                    outcome = ToolOutcome(parse_error, is_error=True)
                else:
                    key = (name, json.dumps(args, sort_keys=True))
                    if key in executed:
                        prior = executed[key]
                        # Replay the original error status. Without this a model
                        # that repeats one *failing* call N times scores N
                        # successes and slips past the guard below.
                        # `item` stays unset so no duplicate item_added fires.
                        outcome = ToolOutcome(
                            "You already made this exact call this turn and it "
                            f"{'failed' if prior.is_error else 'succeeded'}. "
                            f"Previous result: {prior.text}",
                            is_error=prior.is_error,
                        )
                    else:
                        outcome = execute(
                            name, args or {}, state, self.store, timeout=20.0
                        )
                        executed[key] = outcome

                if name == "add_item":
                    if outcome.is_error:
                        adds_failed += 1
                    else:
                        adds_ok += 1

                yield Event(
                    type="tool_result",
                    tool=name,
                    result=outcome.text,
                    is_error=outcome.is_error,
                    iteration=iteration,
                )
                if outcome.item is not None:
                    yield Event(type="item_added", item=outcome.item)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": outcome.text,
                    }
                )
        else:
            yield Event(
                type="warning",
                text=(
                    f"Stopped after {self.settings.max_iterations} rounds. "
                    "The list so far is saved."
                ),
            )

        if final_text:
            state.chat.append(
                ChatTurn(role="assistant", content="\n".join(final_text))
            )

        self.store.save(state)
        yield Event(type="state", state=state)
        yield Event(type="done")
