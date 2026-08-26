# Smart Packing List Agent

An agent that builds a packing list from the **actual forecast** for your trip, not
from a template. It looks up real weather, reasons about what those numbers mean for
your trip type, and justifies every item with a figure it looked up.

```
you> 5 days in Reykjavik starting 2026-08-27, hiking trip

  [call] set_trip(city='Reykjavik', trip_type='hiking', days=5, ...)
  [call] check_weather(city='Reykjavik')
  -> 2026-08-27: 9 to 13C, precip 71% / 4.2mm, wind 41km/h, light rain
     ...
  [+] waterproof jacket  [outerwear]
  [+] hiking boots       [footwear]
  [+] fleece jacket      [layers]
```

## Why this is an agent, not a script

The distinction the design turns on: **thin tools, thick model.**

- `check_weather` returns numbers — temperatures, precipitation probability, wind.
  It never returns a verdict like `"cold and wet"`, and `Forecast` has no
  `needs_raincoat` field. The moment that object holds a conclusion, the tool is
  doing the reasoning and the model is only formatting it.
- `add_item` records a decision. It does not check whether a raincoat makes sense
  in Reykjavik.
- The system prompt names ten **categories** and deliberately never names an
  example item. Listing items would smuggle back in the fixed list this is meant
  to avoid.
- Sequencing is taught, not forced: `add_item` refuses while no forecast is on
  record, and says so *in the tool result*. Ordering stays a model decision rather
  than something `tool_choice` imposes.

So `waterproof jacket` for Iceland is a real inference from `precip 71%`, and the
same agent given a dry forecast will not pack one.

## Setup

Requires Python 3.10+ and a running model server.

```bash
pip install -r requirements.txt
cp .env.example .env
python cli.py
```

Default config points at [Ollama](https://ollama.com) on localhost:

```bash
ollama pull llama3.1:8b
```

## Swapping the model

Everything lives in `.env` — nothing downstream hardcodes an endpoint or a model
name, so a swap is a config edit and no code change.

```ini
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
MODEL=llama3.1:8b
```

| Target | `OPENAI_BASE_URL` | `MODEL` |
| --- | --- | --- |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1:8b` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.1-70b-instruct` |
| llama.cpp / vLLM / LM Studio | whatever it serves on | its model id |

Any OpenAI-compatible endpoint works. The one requirement is **tool-calling
support** — the agent has no text-parsing fallback path for models that lack it
(though it does detect and correct a model that *writes* a tool call as prose).

Other knobs: `TEMPERATURE`, `MAX_TOKENS`, `REQUEST_TIMEOUT`, `MAX_ITERATIONS`
(safety cap on rounds per turn), `HISTORY_TURNS` (chat turns replayed per
request), `DATA_DIR`.

## Running local models: two gotchas

Both of these were found the hard way, and both are environment behaviour rather
than bugs in this code.

**1. Ollama ignores `num_ctx` from the OpenAI-compat endpoint.** On Ollama
0.32.14, `extra_body={"options": {"num_ctx": 16384}}` is silently dropped — both
nested and top-level — so you are stuck at a 4096-token window. Verify with
`ollama ps`, which prints the context actually in use. The architecture is built
to work *within* 4096 tokens (see "Memory" below), so this is survivable. To lift
it anyway, bake it into a derived model:

```bash
ollama create packing-llama -f Modelfile
```

then set `MODEL=packing-llama`.

**2. Pin `keep_alive`, or slow turns compound.** Ollama unloads a model after 5
idle minutes by default. A turn with a dozen tool rounds can cross that boundary,
and then *every* round pays a cold 5.6 GB reload. Measured on one run: 1793s
total. With the model pinned, the same scenario ran at 6–13s per round.

```bash
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","keep_alive":"60m"}'
```

Or set `OLLAMA_KEEP_ALIVE=60m` in the server's environment.

## Commands

| Command | Effect |
| --- | --- |
| *(free text)* | talk to the agent |
| `/check` | **forgot something?** audit the list against trip + forecast |
| `/list` | the packing list, grouped by category |
| `/state` | everything on record, exactly as the model sees it |
| `/trips`, `/use <name>` | multiple trips, one JSON file each |
| `/reset` | delete this trip's data |
| `/help`, `/quit` | |

`python cli.py --trip iceland "5 days in Reykjavik, hiking"` seeds the first
message and selects a trip file.

## Architecture

```
config.py     .env -> frozen Settings
models.py     Pydantic: Trip, Forecast, Item, TripState, Event
store.py      JSON persistence + render_state_block()
weather.py    Open-Meteo client (geocode + forecast, no API key)
tools.py      schemas + implementations
engine.py     the agentic loop; yields Events
cli.py        renders Events to a terminal
```

Strictly layered — each file imports only from those above it. The CLI renders
the engine's event stream directly in the terminal.

### Memory: store-as-memory, not transcript-as-memory

Trip state is re-read from disk and re-rendered into the prompt **every turn** by
`render_state_block()`. Correctness therefore depends on the store, not on the
conversation surviving. That is what makes the agent work in a 4096-token window
and remember across process restarts: a new run of `cli.py` picks up exactly where
the last one left off. `HISTORY_TURNS` of chat ride along for tone, but nothing
load-bearing lives there.

### The "forgot something?" check

`TripState.empty_categories()` is set arithmetic over `CATEGORIES` — pure code.
So `/check` is **deterministic about which categories to examine** and
**model-driven about what belongs in them**. The review prompt explicitly permits
leaving a gap: padding the list is worse than an honest omission.

### Small-model guardrails

An 8B model misbehaves in routine, specific ways. Each is caught and fed back as a
readable correction rather than raised:

- **Improvised argument names.** llama3.1:8b sent `dimension` instead of
  `category` on ten consecutive calls and never recovered. `normalize_args()` maps
  synonyms, and errors now name the offending key. The model still chooses every
  *value*, which is where the reasoning lives.
- **Tool calls written as prose** (`<function=...>`, `<|python_tag|>`) — detected
  and nudged back to the tool interface.
- **Hallucinated success** — the dangerous failure is not an errored call, it is
  the model summarising items it never saved. If every `add_item` failed, the loop
  refuses the final answer and makes it redo them.
- Malformed/double-encoded JSON, duplicate identical calls, empty responses, and a
  hard iteration cap.

Weather failures follow the same rule: `weather.py` never raises to the caller. A
DNS failure comes back as a sentence the model reads and reports honestly — it
does not invent a forecast, and it does not crash the turn.
### Scaling past one process

Swap `Store` for a database-backed class with the same four methods (`load`,
`save`, `list_trips`, `delete`) and pass it in — the engine only depends on that
interface. At that point also move the lock to the database (row lock or
`SELECT … FOR UPDATE`) and the transcript to Redis, since both are currently
per-process.

