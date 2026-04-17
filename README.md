# EU Policy Intelligence Agent

An end-to-end LLMOps project built on Databricks that produces an AI agent capable of answering complex natural language questions over EU legislation.

> **Course project** — LLMOps on Databricks · Updated weekly.

---

## What it does

The agent's knowledge base consists of 7 major EU regulatory documents ingested from PDF, chunked, embedded, and indexed in a Mosaic AI Vector Search index. Users can ask questions like:

- *What obligations does the AI Act impose on high-risk system providers?*
- *How does GDPR define personal data and what are the processing principles?*
- *If I'm building an AI product in the EU, which regulations apply to me?*
- *What does the Digital Markets Act require from gatekeeper platforms?*

The agent uses retrieval-augmented generation (RAG) with tool calling over Databricks Vector Search, persistent multi-turn memory via Lakebase, and is fully traced and evaluated through MLflow. Deployment, promotion, and CI/CD are managed through Databricks Asset Bundles across three isolated environments.

---

## Why this matters

EU digital regulation has accelerated significantly — the AI Act, DSA, DMA, NIS2, Data Act, and Data Governance Act together form a complex and interrelated regulatory landscape. Legal and compliance teams, product managers, and engineers building AI-powered products in Europe need fast, reliable access to this information. This agent bridges that gap by grounding responses in the actual legislative text rather than an LLM's parametric knowledge.

---

## Tech stack

| Layer | Technology |
|---|---|
| Platform | Databricks (Serverless Compute v4) |
| Storage & Governance | Unity Catalog — `dev`, `acc`, `prd` |
| Deployment | Databricks Asset Bundles (DABs) |
| CI/CD | GitHub Actions — matrix deploy to dev + acc on push to `main` |
| Local Development | Databricks Connect + VS Code Databricks extension |
| Package Manager | [uv](https://github.com/astral-sh/uv) |
| Python Packaging | `src/` layout, `pyproject.toml`, built as `.whl` |
| Agent Framework | MLflow `ResponsesAgent` (OpenAI Responses API) |
| Tool Integration | Databricks MCP (Vector Search, optional Genie) |
| Vector Search | Mosaic AI Vector Search |
| Model Serving | Mosaic AI Model Serving — `agents.deploy()` |
| Agent Memory | Lakebase (managed PostgreSQL) via psycopg3 |
| Experiment Tracking | MLflow — tracing, evaluation, model registry |
| Logging | loguru |

---

## Unity Catalog structure

The same schema and volume structure is replicated across all three environments. PDFs must be uploaded to the volume before running the ingestion pipeline.

```
{dev|acc|prd}
└── eu_policy                        ← schema
    ├── legislation                  ← volume (PDFs)
    │   ├── ai_act.pdf
    │   ├── gdpr.pdf
    │   ├── digital_markets_act.pdf
    │   ├── digital_services_act.pdf
    │   ├── nis2_directive.pdf
    │   ├── data_act.pdf
    │   └── data_governance_act.pdf
    ├── raw_documents                ← table: PDF metadata
    ├── ai_parsed_docs               ← table: parsed JSON
    ├── eu_policy_chunks             ← table: clean text chunks + CDF
    └── eu_policy_index              ← Vector Search index
```

Environment promotion managed by the CD pipeline:

```
dev  ──►  acc  ──►  prd
```

---

## Project structure

```
eu-policy-agent/
├── .github/
│   └── workflows/
│       ├── ci.yml                        # PR gate: pre-commit, pytest, uv build
│       └── cd.yml                        # Push to main: matrix deploy to dev + acc
├── notebooks/                            # Databricks notebooks — one per deliverable
│   ├── 1.1_foundation_models_overview.py
│   ├── 1.2_provisioned_throughput_deployment.py
│   ├── 1.3_eu_policy_data_ingestion.py      ← Week 1
│   ├── 1.4_external_models_custom_provider.py
│   ├── 2.2_pdf_parsing_ai_parse.py          ← Week 2
│   ├── 2.3_chunking_strategies.py           ← Week 2
│   ├── 2.4_embeddings_vector_search.py      ← Week 2
│   ├── 3.1_agent_tool_calling.py            ← Week 3 (core)
│   ├── 3.1b_simple_rag.py                  ← Week 3 (supplementary: RAG pattern)
│   ├── 3.2_session_memory_lakebase.py       ← Week 3 (core)
│   ├── 3.3_mcp_integration.py              ← Week 3 (supplementary: MCP deep dive)
│   ├── 3.4_genie_space.py                  ← Week 3 (supplementary: Genie setup)
│   ├── 4.1_mlflow_tracing.py               ← Week 4 (core)
│   ├── 4.2_evaluation.py                   ← Week 4 (core)
│   ├── 4.3_mlflow_log_register.py          ← Week 4
│   ├── 5.1_endpoint_deployment.py          ← Week 5: deploy via agents.deploy() + smoke test
│   └── 5.2_spn_permissions.py              ← Week 5: grant SPN access to serving resources
├── src/
│   └── eu_policy_agent/              # Python package
│       ├── __init__.py
│       ├── config.py                 # Pydantic config + env resolution
│       ├── data_processor.py         # PDF parsing, chunking, Delta writes
│       ├── vector_search.py          # Vector Search endpoint, index, sync, search
│       ├── mcp.py                    # MCP tool integration (ToolInfo, create_mcp_tools)
│       ├── memory.py                 # Lakebase session memory (LakebaseMemory)
│       ├── agent.py                  # EuPolicyAgent + log_register_agent
│       ├── evaluation.py             # Scorers + evaluate_agent runner
│       └── utils/
│           └── common.py             # get_widget, set_mlflow_tracking_uri helpers
├── resources/
│   ├── eu_policy_ingestion_job.yml       # DABs job — PDF ingestion
│   ├── process_data.yml                  # DABs job — data processing + index sync
│   ├── register_deploy_agent.yml         # DABs job — evaluate, log, register, deploy
│   └── deployment_scripts/
│       ├── process_data.py               # Task script: data pipeline
│       ├── log_register_agent.py         # Task script: quality gate → log → register
│       └── deploy_agent.py               # Task script: agents.deploy()
├── tests/
│   ├── conftest.py                  # Shared fixtures, pyspark and SDK stubs
│   ├── test_basic.py                # Package smoke tests
│   └── unit/                        # Unit tests (no live cluster required)
│       ├── test_config.py
│       ├── test_data_processor.py
│       ├── test_vector_search.py
│       ├── test_mcp.py
│       ├── test_memory.py
│       └── test_evaluation.py
├── agent_serving.py                 # MLflow model serving entry point
├── eval_inputs.txt                  # Evaluation question set (33 EU legislation questions)
├── project_config.yml               # Per-environment config (catalog, schema, endpoints)
├── databricks.yml                   # DABs bundle definition (dev / acc / prd targets)
├── pyproject.toml                   # Dependencies + build config
└── version.txt
```

---

## Weekly roadmap

| Week | Deliverable | Status |
|---|---|---|
| 1 | Environment setup · PDF ingestion into `raw_documents` Delta table | ✅ Done |
| 2 | PDF parsing · Chunking · Embeddings · Vector Search index · Data pipeline DABs job | ✅ Done |
| 3 | Agent definition · Tool calling via MCP · Session memory with Lakebase · RAG pattern · MCP deep dive · Genie Space setup | ✅ Done |
| 4 | MLflow tracing · Agent evaluation · Model logging and registration to Unity Catalog | ✅ Done |
| 5 | Agent deployment via `agents.deploy()` · CI/CD pipeline · SPN permissions | ✅ Done |
| 6 | Monitoring & observability · Trace aggregation · AIBI dashboards · Alerting · FinOps | ⬜ Planned |

---

## Setup

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) — `pip install uv`
- [Databricks CLI v0.200+](https://docs.databricks.com/dev-tools/cli/install.html)
- [VS Code Databricks extension](https://marketplace.visualstudio.com/items?itemName=databricks.databricks)

### 1. Clone and install

```bash
git clone https://github.com/hugodscarvalho/eu-policy-agent.git
cd eu-policy-agent
uv sync --extra dev
```

### 2. Authenticate with Databricks

```bash
databricks configure --host https://<your-workspace-url>
```

Or use the VS Code Databricks extension — it configures `databricks-connect` automatically.

### 3. Configure `project_config.yml`

Fill in the fields marked with comments. Required fields must be set before running any notebook.

```yaml
dev:
  catalog: dev
  schema: eu_policy
  volume: legislation
  llm_endpoint: databricks-llama-4-maverick       # required
  embedding_endpoint: databricks-gte-large-en     # required
  warehouse_id: "520a7ce3b05d3359"                # required for Genie
  vector_search_endpoint: eu_policy_vs_endpoint   # required
  genie_space_id: ""           # optional — leave empty to disable Genie tooling
  lakebase_project_id: ""      # optional — leave empty to run agent without memory
  usage_policy_id: ""          # optional — Databricks serverless usage policy ID
  experiment_path: "/Users/you@example.com/eu-policy-agent-dev"  # required for Week 4+
```

See the [Configuration](#configuration) section for full field reference.

### 4. Create Unity Catalog objects

```sql
-- Run in Databricks SQL or a notebook
CREATE CATALOG IF NOT EXISTS dev;
CREATE SCHEMA IF NOT EXISTS dev.eu_policy;
CREATE VOLUME IF NOT EXISTS dev.eu_policy.legislation;
-- Repeat for acc and prd
```

Then upload the 7 PDF files (from `materials/week1/eu-legislation/`) to each volume:
- `/Volumes/dev/eu_policy/legislation/`
- `/Volumes/acc/eu_policy/legislation/`
- `/Volumes/prd/eu_policy/legislation/`

### 5. Deploy the bundle

```bash
# Deploy to dev (default)
databricks bundle deploy

# Deploy to a specific environment
databricks bundle deploy --target acc
```

---

## Running the project — week by week

Each week builds on the previous one. Run notebooks in the Databricks UI after deploying the bundle, or trigger them as jobs from the CLI.

### Week 1 — PDF ingestion

**Notebook:** `notebooks/1.3_eu_policy_data_ingestion.py`

Lists PDFs from the volume, extracts page counts, and writes metadata to `raw_documents`.

```bash
databricks bundle run eu_policy_ingestion_job
```

Also available as a standalone notebook in the Databricks UI.

---

### Week 2 — Parsing, chunking, and Vector Search

**Notebooks (run in order):**
1. `notebooks/2.2_pdf_parsing_ai_parse.py` — parses PDFs with `ai_parse_document`, writes JSON to `ai_parsed_docs`
2. `notebooks/2.3_chunking_strategies.py` — extracts and cleans text chunks, writes to `eu_policy_chunks`
3. `notebooks/2.4_embeddings_vector_search.py` — creates Vector Search endpoint and index, demonstrates semantic / hybrid / reranked search

Or run the full pipeline as a scheduled DABs job:

```bash
databricks bundle run data-pipeline
```

The job runs daily at 06:00 Lisbon time in production (paused by default in dev/acc).

---

### Week 3 — Agent definition, tool calling, and session memory

> **Bundle deploy required?** No. All Week 3 notebooks run interactively on Databricks compute or locally via Databricks Connect. No `databricks bundle deploy` is needed.

**Core notebooks (required):**

#### `notebooks/3.1_agent_tool_calling.py` — Agent and MCP tools

Covers:
- MCP tool discovery from the Vector Search endpoint
- `ToolInfo` spec format (OpenAI Responses API)
- Direct tool execution smoke test
- `EuPolicyAgent` initialisation and single-turn query demo
- MCP vs custom function trade-offs

**Prerequisites:**
- Weeks 1 and 2 complete (Vector Search index populated)
- `llm_endpoint` set in `project_config.yml`

**Optional:** `genie_space_id` — if empty, the Genie MCP tool is skipped and the agent runs with Vector Search only.

---

#### `notebooks/3.2_session_memory_lakebase.py` — Session memory with Lakebase

Covers:
- Provisioning a Lakebase project (PostgreSQL cluster)
- Creating the `session_messages` table
- `LakebaseMemory` save and load
- Multi-turn conversation demo with memory

**Prerequisites:**
- Lakebase must be enabled in your Databricks workspace
- `lakebase_project_id` must be set in `project_config.yml`

> **Optional overall**: The agent runs without memory when `lakebase_project_id` is empty. Run this notebook only if you want multi-turn conversation support.

**Production authentication:** In Model Serving, Lakebase must be accessed via a Service Principal. Set these environment variables on your serving endpoint:

```
LAKEBASE_SP_CLIENT_ID      = <service-principal-client-id>
LAKEBASE_SP_CLIENT_SECRET  = <service-principal-secret>
LAKEBASE_SP_HOST           = https://<workspace>.azuredatabricks.net
```

`LakebaseMemory` detects these automatically — no code changes needed.

---

**Supplementary notebooks (deepen understanding):**

#### `notebooks/3.1b_simple_rag.py` — Simple RAG pattern

Covers the deterministic RAG alternative to full agent tool-calling: direct Vector Search retrieval → prompt injection → LLM response. Includes a `SimpleRAG` class with multi-turn conversation history. Useful for understanding the retrieval-first approach before moving to the full agentic loop.

**Prerequisites:** Same as `3.1_agent_tool_calling.py`.

---

#### `notebooks/3.3_mcp_integration.py` — MCP deep dive

Goes deeper into the MCP protocol using `DatabricksMCPClient` directly before the `eu_policy_agent` package abstracts it away. Demonstrates raw tool listing, direct tool calls, multi-server tool registries, and a `SimpleAgent` class that shows the core tool-calling loop.

**Prerequisites:** Same as `3.1_agent_tool_calling.py`.

---

#### `notebooks/3.4_genie_space.py` — Genie Space setup

Creates (or verifies) a Databricks Genie Space connected to `eu_policy_chunks` and `raw_documents`. Tests natural language → SQL conversion. After running, copy the Space ID into `genie_space_id` in `project_config.yml` to enable the Genie MCP tool in `EuPolicyAgent`.

**Prerequisites:**
- `warehouse_id` set in `project_config.yml`
- Genie enabled in your Databricks workspace

> Run this before `4.3_mlflow_log_register.py` if you want `DatabricksGenieSpace` declared in the model resources.

---

### Week 4 — MLflow tracing, evaluation, and registration

> **Bundle deploy required?** No. All Week 4 notebooks run interactively on Databricks compute or locally via Databricks Connect. No `databricks bundle deploy` is needed.

**Notebooks (run in order):**

#### `notebooks/4.1_mlflow_tracing.py` — Tracing

Covers:
- `@mlflow.trace` decorator and `SpanType` hierarchy (AGENT / CHAIN / LLM / TOOL / RETRIEVER)
- `mlflow.start_span` for manual span control
- `mlflow.update_current_trace` for attaching `session_id`, `git_sha`, and deployment metadata
- Trace search by session ID and git SHA
- Performance analysis across recent traces

**Prerequisites:**
- `experiment_path` set in `project_config.yml`
- Week 3 agent working (Vector Search index populated)

---

#### `notebooks/4.2_evaluation.py` — Evaluation suite

Covers:
- **Guidelines** (LLM-as-judge, binary): `polite_tone`, `stays_in_scope`, `cites_regulation`
- **Code-based scorers** (deterministic): `word_count_check` (< 500 words), `mentions_legislation`, `response_length_score` (0–1 float)
- **Numeric judge**: `response_quality` scored 1–5 via `make_judge`
- **Categorical judge**: `scope_classification` (in_scope / partially_in_scope / out_of_scope)
- Combined evaluation using `eval_inputs.txt` (33 EU legislation questions)

**Prerequisites:**
- `experiment_path` and `llm_endpoint` set in `project_config.yml`
- `eval_inputs.txt` at the repo root (already present)

**Cost note:** Guidelines and judge scorers call the LLM once per row. Use code-based scorers (`word_count_check`, `mentions_legislation`) for fast iterative runs; add LLM scorers for pre-registration quality gates.

---

#### `notebooks/4.3_mlflow_log_register.py` — Log and register to Unity Catalog

Covers:
- Running evaluation as a quality gate before logging
- `mlflow.pyfunc.log_model` with resource declarations (Vector Search index, tables, endpoints, SQL warehouse, and optionally Genie Space)
- `mlflow.register_model` to Unity Catalog
- Setting the `latest-model` alias for deployment

**Prerequisites:**
- All of the above weeks complete
- `experiment_path` set in `project_config.yml`
- The `agent_serving.py` file at the repo root (already present — do not rename)

> **Note on `agent_serving.py`:** The model serving entry point is deliberately named `agent_serving.py`, not `eu_policy_agent.py`. A top-level file matching the package name would shadow the `eu_policy_agent` package on `sys.path` and break all imports.

> **Genie resource:** If `genie_space_id` is set, `DatabricksGenieSpace` is automatically added to the resource declarations. This grants the Model Serving service identity access to the Genie Space — no manual permission setup needed.

---

### Week 5 — Agent deployment and CI/CD

**Prerequisites:**
- Week 4 complete — a registered model with the `latest-model` alias set
- A secret scope `eu-policy-agent-scope` in Databricks with `client-id` and `client-secret` for the Lakebase SPN

#### `notebooks/5.2_spn_permissions.py` — Grant SPN permissions *(run first)*

Before the serving endpoint can reach workspace resources, the Model Serving service identity must have permission to access them. This notebook grants:

| Resource | Permission |
|---|---|
| Genie Space | `CAN_RUN` |
| Vector Search endpoint | `CAN_USE` |
| SQL Warehouse | `CAN_USE` |

Reads the SPN `client_id` from the `{env}_SPN` Databricks secret scope. Run once per environment as a workspace admin.

---

#### `notebooks/5.1_endpoint_deployment.py` — Deploy and test the endpoint

Covers:
- `agents.deploy()` — endpoint creation, service-identity provisioning, inference tables, review app
- Passing `MLFLOW_EXPERIMENT_ID` so production traces land in the correct experiment
- Lakebase SPN credentials injected as `{{secrets/...}}` references (not hard-coded)
- Scale-to-zero and workload sizing
- Testing the live endpoint via the OpenAI Responses API client

**Prerequisites:**
- `notebooks/5.2_spn_permissions.py` executed for the target environment
- `experiment_path` set in `project_config.yml`
- Secret scope `eu-policy-agent-scope` populated

> **Authentication note:** Lakebase credentials are injected via `LAKEBASE_SP_CLIENT_ID`, `LAKEBASE_SP_CLIENT_SECRET`, `LAKEBASE_SP_HOST` environment variables — not through `DATABRICKS_CLIENT_*`. Using `DATABRICKS_CLIENT_*` on the serving endpoint overrides MCP resource auth and breaks Vector Search tool calls.

---

#### CI/CD pipeline

The automated pipeline runs entirely through GitHub Actions and Databricks Asset Bundles.

**CI** (`.github/workflows/ci.yml`) — triggered on every pull request to `main`:

```
pre-commit (ruff + standard hooks)  →  pytest  →  uv build
```

**CD** (`.github/workflows/cd.yml`) — triggered on push to `main`:

```
matrix: [dev, acc]
  └── databricks bundle deploy --target {env} --var="git_sha=..." --var="branch=..."
```

The bundle deploy triggers the `eu-policy-agent-register-deploy-pipeline` Lakeflow job, which runs two tasks in sequence:

```
log_register_agent  →  deploy_agent
```

1. **`log_register_agent`** — runs evaluation as a quality gate, logs the model to MLflow, registers it to Unity Catalog, and sets the `latest-model` alias.
2. **`deploy_agent`** — resolves the `latest-model` alias version and calls `agents.deploy()` to update the Model Serving endpoint.

**Required GitHub setup (one-time, per repo):**

Create two GitHub Environments (`dev`, `acc`) in repo Settings → Environments, each with:

| Key | Type | Value |
|---|---|---|
| `DATABRICKS_HOST` | Variable | `https://<your-workspace>.azuredatabricks.net` |
| `DATABRICKS_CLIENT_ID` | Secret | SPN client ID for the environment |
| `DATABRICKS_CLIENT_SECRET` | Secret | SPN client secret for the environment |

---

## Development

### Testing

Unit tests run without a live Databricks cluster — all external dependencies are stubbed.

```bash
# Install ci dependencies
uv sync --extra ci

# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Specific module
uv run pytest tests/unit/test_config.py
uv run pytest tests/unit/test_memory.py
uv run pytest tests/unit/test_evaluation.py
```

119 tests, ~0.3s.

---

### Linting and formatting

Ruff is not a global binary in this project — always invoke it via `uv run --with ruff`:

```bash
# Lint and auto-fix
uv run --with ruff ruff check . --fix

# Format
uv run --with ruff ruff format .
```

---

### Pre-commit hooks

Pre-commit runs ruff (lint + format) and standard checks on every `git commit`.

```bash
# One-time setup
uv run pre-commit install

# Run against all files manually
uv run pre-commit run --all-files
```

Once installed, hooks run automatically on `git commit`. If a hook auto-fixes a file, the commit is blocked — `git add` the fixed files and retry.

---

## Configuration

`project_config.yml` drives all environment-specific values. The active section is resolved at runtime from the `env` Databricks widget (set by the bundle target), falling back to `dev` for local development.

### Field reference

| Field | Required | Description |
|---|---|---|
| `catalog` | Yes | Unity Catalog catalog name |
| `schema` | Yes | Schema name (always `eu_policy`) |
| `volume` | Yes | Volume name for PDFs (always `legislation`) |
| `llm_endpoint` | Yes | Databricks model serving endpoint for the LLM |
| `embedding_endpoint` | Yes | Databricks model serving endpoint for embeddings |
| `warehouse_id` | Yes | SQL warehouse ID (used by Genie) |
| `vector_search_endpoint` | Yes | Vector Search endpoint name |
| `genie_space_id` | Optional | Genie Space ID for natural language data queries. Leave empty to disable. |
| `lakebase_project_id` | Optional | Lakebase project ID for session memory. Leave empty to run the agent stateless. |
| `usage_policy_id` | Optional | Databricks serverless usage policy ID for cost attribution and chargeback. |
| `experiment_path` | Yes (Week 4+) | MLflow experiment path, e.g. `/Shared/eu-policy-agent-dev` |
| `system_prompt` | No | Agent system prompt. Defaults to the EU legislation QA prompt. |

```yaml
dev:
  catalog: dev
  schema: eu_policy
  volume: legislation
  llm_endpoint: databricks-llama-4-maverick
  embedding_endpoint: databricks-gte-large-en
  warehouse_id: "520a7ce3b05d3359"
  vector_search_endpoint: eu_policy_vs_endpoint
  genie_space_id: ""           # optional
  lakebase_project_id: ""      # optional
  usage_policy_id: ""          # optional
  experiment_path: "/Shared/eu-policy-agent-dev"
```

---

## Manual setup checklist

Steps that cannot be automated and must be done once per environment before running the pipeline end-to-end:

- [ ] Upload the 7 EU legislation PDFs to `/Volumes/{env}/eu_policy/legislation/`
- [ ] Create the `eu-policy-agent-scope` Databricks secret scope with `client-id` and `client-secret`
- [ ] Create the `{env}_SPN` secret scope with `client_id` (used by `5.2_spn_permissions.py`)
- [ ] Run `notebooks/5.2_spn_permissions.py` as a workspace admin for each environment
- [ ] Create GitHub Environments `dev` and `acc` with `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`
- [ ] Fill in `genie_space_id` and `lakebase_project_id` in `project_config.yml` after provisioning those resources
- [ ] Fill in `usage_policy_id` in `project_config.yml` once policy IDs are assigned in the workspace

---

## Git conventions

| Convention | Format |
|---|---|
| Branch | `week{n}/{short-description}` or `week{n}-{m}/{short-description}` for multi-week spans |
| PR title | `[Week N] Description` — e.g. `[Week 5 & 6] Deployment, CI/CD, and observability` |
| Direct commits to `main` | Never |

All changes go through a PR. `main` always reflects the latest stable weekly deliverable.

---

## Author

**Hugo Carvalho** · [github.com/hugodscarvalho](https://github.com/hugodscarvalho)
