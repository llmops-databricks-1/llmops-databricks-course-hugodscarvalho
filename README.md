# EU Policy Intelligence Agent

An end-to-end LLMOps project built on Databricks that produces an AI agent capable of answering complex natural language questions over EU legislation.

> **Course project** - LLMOps on Databricks · Work in progress, updated weekly.

---

## What it does

The agent's knowledge base consists of 7 major EU regulatory documents ingested from PDF, chunked, embedded, and indexed in a Mosaic AI Vector Search index. Users can ask questions like:

- *What obligations does the AI Act impose on high-risk system providers?*
- *How does GDPR define personal data and what are the processing principles?*
- *If I'm building an AI product in the EU, which regulations apply to me?*
- *What does the Digital Markets Act require from gatekeeper platforms?*

The agent uses retrieval-augmented generation (RAG) with tool calling, persistent memory via Lakebase, and is fully traced and evaluated through MLflow. Deployment, promotion, and CI/CD are managed through Databricks Asset Bundles across three isolated environments.

---

## Why this matters

EU digital regulation has accelerated significantly - the AI Act, DSA, DMA, NIS2, Data Act, and Data Governance Act together form a complex and interrelated regulatory landscape. Legal and compliance teams, product managers, and engineers building AI-powered products in Europe need fast, reliable access to this information. This agent bridges that gap by grounding responses in the actual legislative text rather than an LLM's parametric knowledge.

---

## Tech stack

| Layer | Technology |
|---|---|
| Platform | Databricks (Serverless Compute v4) |
| Storage & Governance | Unity Catalog - `dev`, `acc`, `prd` |
| Deployment | Databricks Asset Bundles (DABs) |
| Local Development | Databricks Connect + VS Code Databricks extension |
| Package Manager | [UV](https://github.com/astral-sh/uv) |
| Python Packaging | `src/` layout, `pyproject.toml`, built as `.whl` |
| Vector Search | Mosaic AI Vector Search |
| Model Serving | Mosaic AI Model Serving |
| Agent Memory | Lakebase |
| Experiment Tracking | MLflow (tracing, evaluation, prompt registry) |
| Logging | loguru |

---

## Unity Catalog structure

The same schema and volume structure is replicated across all three environments. PDFs must be uploaded to the volume before running the ingestion notebook.

```
{dev|acc|prd}
└── eu_policy                        ← schema
    └── legislation                  ← volume (PDFs)
        ├── ai_act.pdf
        ├── gdpr.pdf
        ├── digital_markets_act.pdf
        ├── digital_services_act.pdf
        ├── nis2_directive.pdf
        ├── data_act.pdf
        └── data_governance_act.pdf
```

Environment promotion flow managed by DABs CI/CD (built in Week 6):

```
dev  ──►  acc  ──►  prd
```

---

## Project structure

```
llmops-databricks-course-hugodscarvalho/
├── notebooks/                  # Databricks notebooks (one per deliverable)
├── src/
│   └── eu_policy_agent/        # Python package
│       ├── __init__.py
│       └── config.py           # Pydantic config + env resolution
├── resources/                  # DABs job/pipeline YAML definitions
├── project_config.yml          # Per-environment config (catalog, schema, endpoints)
├── databricks.yml              # DABs bundle definition (dev / acc / prd targets)
├── pyproject.toml              # Dependencies + build config
└── version.txt
```

---

## Weekly roadmap

| Week | Deliverable | Status |
|---|---|---|
| 1 | Environment setup · PDF ingestion into Delta tables (`raw_documents`) | 🔄 In progress |
| 2 | Chunking · Embeddings · Vector Search index · Genie Space | ⬜ Planned |
| 3 | Agent definition · Tool calling · Memory with Lakebase | ⬜ Planned |
| 4 | MLflow tracing · Evaluation · Prompt optimisation | ⬜ Planned |
| 5 | Agent deployment · Monitoring and observability | ⬜ Planned |
| 6 | CI/CD pipeline via DABs · Promotion `dev → acc → prd` | ⬜ Planned |

---

## Setup

### Prerequisites

- Python 3.12
- [UV](https://github.com/astral-sh/uv) - `pip install uv`
- [Databricks CLI v0.200+](https://docs.databricks.com/dev-tools/cli/install.html)
- [VS Code Databricks extension](https://marketplace.visualstudio.com/items?itemName=databricks.databricks)

### 1. Clone and install

```bash
git clone https://github.com/hugodscarvalho/llmops-databricks-course-hugodscarvalho.git
cd llmops-databricks-course-hugodscarvalho

uv sync --extra dev
```

### 2. Authenticate with Databricks

```bash
databricks configure --host https://<your-workspace-url>
```

Or use the VS Code Databricks extension to sign in - it will configure `databricks-connect` automatically.

### 3. Configure local environment

Fill in the blank fields in `project_config.yml` (warehouse ID, VS endpoint) and verify `databricks.yml` points to your workspace hosts for `acc` and `prd`.

### 4. Create Unity Catalog objects

Run the following once per environment before executing any notebooks:

```sql
-- Repeat for acc and prd catalogs as needed
CREATE CATALOG IF NOT EXISTS dev;
CREATE SCHEMA IF NOT EXISTS dev.eu_policy;
CREATE VOLUME IF NOT EXISTS dev.eu_policy.legislation;
```

Then upload the 7 PDF files to `/Volumes/dev/eu_policy/legislation/`.

### 5. Deploy the bundle

```bash
# Deploy to dev (default)
databricks bundle deploy

# Deploy to a specific target
databricks bundle deploy --target acc
```

---

## Git conventions

| Convention | Format |
|---|---|
| Branch | `week{n}/{short-description}` - e.g. `week1/databricks-setup-and-ingestion` |
| PR title | `[Week N] Description` - e.g. `[Week 1] PDF ingestion into Delta tables` |
| Direct commits to `main` | Never |

All changes go through a PR. `main` always reflects the latest stable weekly deliverable.

---

## Configuration

`project_config.yml` drives all environment-specific values. The active section is resolved at runtime from the `env` Databricks widget (set by the bundle target), falling back to `dev` for local development via `databricks-connect`.

```yaml
dev:
  catalog: dev
  schema: eu_policy
  volume: legislation
  llm_endpoint: databricks-llama-4-maverick
  embedding_endpoint: databricks-gte-large-en
  ...

acc:
  catalog: acc
  ...

prd:
  catalog: prd
  ...
```

---

## Author

**Hugo Carvalho** · [github.com/hugodscarvalho](https://github.com/hugodscarvalho)


