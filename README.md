# Smart Packing List Agent

An AI-powered packing list agent that builds personalized packing lists using **real-time weather forecasts**, trip context, and model-driven reasoning.

Instead of relying on predefined packing templates, the agent retrieves actual weather data, interprets environmental conditions, and recommends relevant items with evidence-backed explanations.

## Example

**Input:**
```text
5 days in Reykjavik starting 2026-08-27, hiking trip
```

**Agent workflow:**
```text
User Request
    |
    v
set_trip(city="Reykjavik", trip_type="hiking", days=5)
    |
    v
check_weather(city="Reykjavik")
    |
    v
Weather Data
    |
    v
Model-driven reasoning
    |
    v
Personalized Packing List
```

**Example output:**
```text
Forecast:
- Temperature: 9–13°C
- Precipitation probability: 71%
- Expected precipitation: 4.2mm
- Wind speed: 41km/h

Recommended items:
- Waterproof jacket [Outerwear]
- Hiking boots [Footwear]
- Fleece jacket [Layers]
```

Each recommendation is based on retrieved weather information and trip context.

---

## Why This Is an Agent, Not a Script

The core architectural principle is:

**Thin Tools, Thick Model.**

The tools retrieve information and execute actions. The model is responsible for reasoning and decision-making.

### 1. Thin Tools

`check_weather` returns raw weather information such as:
- Temperature
- Precipitation probability
- Wind speed
- Expected rainfall

It does not return conclusions like `"cold and wet"` or predefined recommendations such as `needs_raincoat`.

The model interprets the retrieved information.

### 2. Model-Driven Decisions

`add_item` records a decision. It does not determine whether an item is appropriate for a particular destination.

The model decides what to pack based on:
- Weather conditions
- Trip duration
- Trip type
- Environmental context

### 3. Dynamic Recommendations

The system prompt defines ten packing categories without prescribing example items.

This prevents the agent from behaving like a fixed packing checklist.

### 4. Tool Sequencing

The agent learns the correct execution sequence through tool feedback.

For example, `add_item` refuses to save an item if no forecast has been recorded.

The model must interpret this feedback and decide what to do next.

This keeps reasoning inside the model instead of enforcing the entire workflow through hardcoded tool sequencing.

---

## Features

- **Weather-aware recommendations:** Uses real forecasts from Open-Meteo.
- **Agentic tool calling:** The model decides when and how to invoke tools.
- **Persistent memory:** Trip state is stored in JSON files and restored across sessions.
- **Multi-trip support:** Maintain independent packing lists for different trips.
- **Deterministic testing:** Automated tests use mocked model and HTTP clients.
- **Structured logging:** JSON and text logging with configurable log levels.
- **Observability:** Optional health, readiness, and Prometheus metrics endpoints.
- **Docker support:** Containerized application with persistent storage.
- **CI/CD:** GitHub Actions workflows for testing and Docker builds.
- **Kubernetes manifests:** Single-replica deployment baseline.

---

## Architecture

```text
                 User
                  |
                  v
                 CLI
                  |
                  v
              Engine.py
          (Agentic Execution Loop)
                  |
          +-------+-------+
          |       |       |
          v       v       v
        Model   Tools   Memory
          |       |       |
          |       v       |
          |    Weather    |
          |     Client    |
          |       |       |
          |       v       |
          |  Open-Meteo    |
          |               |
          +-------+-------+
                  |
                  v
              Store.py
                  |
                  v
             JSON Storage
```

### Project Structure

```text
.
├── app/
│   ├── config.py
│   ├── engine.py
│   ├── store.py
│   ├── tools.py
│   ├── health.py
│   └── observability.py
│
├── k8s/
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── cli.py
├── pyproject.toml
├── requirements-dev.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Ollama or another OpenAI-compatible model server
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/rishavraj13/Packing-agent-devops.git

cd Packing-agent-devops
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Default configuration:

```ini
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
MODEL=llama3.1:8b
```

### 4. Start Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Pull the default model:

```bash
ollama pull llama3.1:8b
```

### 5. Run the Agent

```bash
python cli.py
```

---

## Model Configuration

The application uses environment-based configuration.

Switching models does not require modifying application code.

| Provider | OPENAI_BASE_URL | MODEL |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | `llama3.1:8b` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.1-70b-instruct` |
| llama.cpp / vLLM / LM Studio | Provider-specific endpoint | Provider-specific model ID |

Any OpenAI-compatible endpoint with tool-calling support can be used.

The agent does not have a text-parsing fallback for models without tool-calling support.

### Additional Configuration

| Variable | Purpose |
|---|---|
| `TEMPERATURE` | Controls response randomness |
| `MAX_TOKENS` | Maximum response tokens |
| `REQUEST_TIMEOUT` | Model request timeout |
| `MAX_ITERATIONS` | Maximum agent execution rounds |
| `HISTORY_TURNS` | Conversation history retained per request |
| `DATA_DIR` | Persistent storage directory |

---

## Local Model Optimization

Two important issues were identified while running the application with Ollama.

### 1. Context Window Configuration

Ollama may ignore `num_ctx` passed through its OpenAI-compatible endpoint.

On Ollama 0.32.14, passing:

```python
extra_body={"options": {"num_ctx": 16384}}
```

was silently ignored.

The actual context window can be verified using:

```bash
ollama ps
```

The architecture is designed to operate within a 4096-token context window.

To create a model with a larger context configuration, use a custom Modelfile:

```bash
ollama create packing-llama -f Modelfile
```

Then configure:

```ini
MODEL=packing-llama
```

### 2. Model Keep-Alive

Ollama unloads inactive models after five minutes by default.

Long agentic execution loops can cross this threshold, resulting in repeated model reloads.

Keep the model loaded:

```bash
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","keep_alive":"60m"}'
```

Alternatively, configure:

```ini
OLLAMA_KEEP_ALIVE=60m
```

---

## CLI Commands

| Command | Description |
|---|---|
| Free text | Interact with the agent |
| `/check` | Audit the packing list against trip requirements and forecast |
| `/list` | Display items grouped by category |
| `/state` | Display the complete persisted trip state |
| `/trips` | List saved trips |
| `/use <trip_id>` | Switch between saved trips |
| `/reset` | Delete the current trip's data |
| `/help` | Display available commands |
| `/quit` | Exit the application |

### Start a Trip Directly

```bash
python cli.py --trip iceland "5 days in Reykjavik, hiking"
```

This seeds the initial message and selects the corresponding trip file.

New trips started with the default trip ID are saved using a name derived from the destination, trip type, and start date.

Example:

```text
reykjavik-hiking-2026-08-27
```

---

## Memory Architecture

### Store-as-Memory, Not Transcript-as-Memory

The application uses persistent structured state rather than relying entirely on conversation history.

Every turn:

1. Trip state is loaded from disk.
2. `render_state_block()` converts the state into a structured prompt.
3. The model receives the current trip context.
4. Tool calls update the persisted state.
5. The updated state is available in subsequent turns.

This provides:

- Persistence across application restarts.
- Consistent state management.
- Reduced dependence on conversation history.
- Better compatibility with smaller context windows.

`HISTORY_TURNS` of conversation history is retained for conversational continuity, but critical application state lives in persistent storage.

### Deterministic Packing Audit

`TripState.empty_categories()` uses set arithmetic over `CATEGORIES`.

This creates a hybrid approach:

- **Deterministic code:** Determines which categories need review.
- **Model-driven reasoning:** Determines which items belong in those categories.

The review prompt also allows the model to leave a category empty when no additional item is justified.

---

## Agent Reliability and Small-Model Guardrails

Smaller models can produce predictable tool-calling errors.

The application handles these through validation, normalization, and corrective feedback.

| Problem | Handling |
|---|---|
| Incorrect argument names | `normalize_args()` maps known synonyms |
| Tool calls generated as prose | Detects malformed tool-call output and requests correction |
| Hallucinated success | Prevents successful completion when required item additions fail |
| Malformed JSON | Validates and handles malformed or double-encoded arguments |
| Duplicate tool calls | Detects repeated identical calls |
| Empty model responses | Handles empty responses |
| Infinite execution loops | Enforces a maximum iteration count |

### Honest Weather Failure Handling

Weather failures are handled without crashing the agent.

For example, if DNS resolution fails, `weather.py` returns an explanatory error message to the model.

The model can then communicate the failure instead of inventing weather information.

---

## Testing

The automated test suite is deterministic.

It uses:
- Temporary directories for storage.
- Mock model clients.
- Mock HTTP clients.

No running Ollama server, API key, or internet connection is required.

### Run Tests

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run:

```bash
pytest
```

---

## Logging and Observability

### Structured Logging

Operational logs are written to `stderr`, while normal CLI output remains on `stdout`.

Supported configuration:

```ini
LOG_LEVEL=INFO
LOG_FORMAT=json
```

Supported log levels:

- DEBUG
- INFO
- WARNING
- ERROR
- CRITICAL

Supported formats:

- JSON
- Text

Logs deliberately exclude API keys, authorization headers, prompts, and environment values.

### Health and Metrics Endpoints

The optional standard-library HTTP server is disabled by default.

Start observability independently:

```bash
python cli.py --observability-only
```

Or alongside the CLI:

```bash
python cli.py --serve-observability
```

Available endpoints:

| Endpoint | Purpose |
|---|---|
| `/healthz` | Health check |
| `/readyz` | Readiness check |
| `/metrics` | Prometheus-format metrics |

Readiness checks local writable storage. It does not perform model inference or weather requests.

---

## Docker

### Build Image

```bash
docker build -t packing-list-agent:local .
```

### Run Container

```bash
docker run --rm -it \
  --env-file .env \
  -v packing-data:/data \
  packing-list-agent:local
```

`DATA_DIR` defaults to `/data` inside the container.

Mounting a volume ensures trip data persists after the container exits.

### External Model Endpoint

The model endpoint remains external to the container.

For example:

```bash
docker run --rm -it \
  -e OPENAI_BASE_URL=https://model.example/v1 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  -v packing-data:/data \
  packing-list-agent:local
```

Use a proper secret-injection mechanism in production. Never place secrets in Docker image layers.

### Run Observability Mode

```bash
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  packing-list-agent:local \
  --observability-only
```

---

## CI/CD with GitHub Actions

The repository includes automated workflows under `.github/workflows/`.

### Continuous Integration

`ci.yml` runs on:
- Pull requests.
- Pushes to the `main` branch.

It performs:

1. Deterministic automated testing.
2. Docker image build verification.

No production secrets are required for CI.

### Continuous Delivery

The separate publishing workflow:

- Runs on version tags matching `v*`.
- Supports manual dispatch.
- Builds and publishes Docker images to GitHub Container Registry (GHCR).

Example image tags:

```text
ghcr.io/<owner>/<repository>:vX.Y.Z
ghcr.io/<owner>/<repository>:sha-<commit>
```

Repository owners must configure GitHub Actions package write permissions.

Forks generally cannot publish packages using the built-in GitHub Actions token without the appropriate permissions.

The publishing workflow does not deploy infrastructure.

---

## Kubernetes

The `k8s/` directory contains baseline Kubernetes manifests designed for a single-replica deployment.

### Deployment

Before deploying:

1. Configure the container image.
2. Configure non-secret model endpoint settings.
3. Create the required Kubernetes Secret outside source control.
4. Ensure the cluster has outbound connectivity to the model endpoint and Open-Meteo.

### Validate Manifests

Render the configuration:

```bash
kubectl kustomize k8s
```

Validate without applying:

```bash
kubectl apply --dry-run=client -k k8s
```

### Important Considerations

- JSON state requires persistent storage.
- Keep a single replica until shared storage and concurrency control are implemented.
- Kubernetes Secrets are base64-encoded, not encrypted by default.
- Use a dedicated secret manager for production workloads.

Refer to `docs/operations.md` for operational guidance.

---

## Scaling Beyond a Single Process

The current architecture uses a storage abstraction.

The `Store` implementation exposes four methods:

```python
load()
save()
list_trips()
delete()
```

This allows the JSON-backed implementation to be replaced with a database-backed implementation without changing the agent engine's dependency on the storage interface.

For multi-process deployment, additional changes are required:

- Replace process-local locking with database-level locking.
- Use row locks or `SELECT ... FOR UPDATE` for concurrent updates.
- Move conversation transcripts to Redis if shared transcript state is needed.
- Introduce shared persistence and concurrency management.

The current JSON-based implementation is intentionally designed for a single-process deployment.

---

## Configuration Reference

### Model Configuration

| Variable | Description |
|---|---|
| `OPENAI_BASE_URL` | Model server endpoint |
| `OPENAI_API_KEY` | Model API key |
| `MODEL` | Model identifier |
| `TEMPERATURE` | Sampling temperature |
| `MAX_TOKENS` | Maximum output tokens |
| `REQUEST_TIMEOUT` | Model request timeout |
| `MAX_ITERATIONS` | Maximum agent execution rounds |
| `HISTORY_TURNS` | Conversation history retained |
| `DATA_DIR` | Persistent data directory |

### Operational Configuration

| Variable | Description |
|---|---|
| `WEATHER_TIMEOUT` | Weather API request timeout |
| `LOG_LEVEL` | Logging verbosity |
| `LOG_FORMAT` | JSON or text logging |
| `OBSERVABILITY_ENABLED` | Enable observability server |
| `OBSERVABILITY_HOST` | Observability server host |
| `OBSERVABILITY_PORT` | Observability server port |

---

## DevOps / SRE Implementation

This project extends an agentic AI application with a practical DevOps and operational baseline.

| Area | Implementation |
|---|---|
| Testing | Deterministic pytest suite |
| Logging | Structured JSON and text logging |
| Monitoring | Prometheus-compatible metrics |
| Health Checks | Health and readiness endpoints |
| Containerization | Docker |
| CI | GitHub Actions |
| Container Registry | GitHub Container Registry |
| Orchestration | Kubernetes |
| Persistence | JSON-backed storage with persistent volumes |
| Configuration | Environment-based settings |

---

## Future Improvements

Potential next steps include:

- Database-backed persistent storage.
- Redis-based shared state.
- Multi-replica deployment support.
- Improved observability and tracing.
- Automated deployment pipelines.
- More advanced agent evaluation and reliability testing.

---

## License

Add your preferred open-source license before distributing this project.

