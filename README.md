# FixIt AI Support System — LLMOps

> **Version:** 1.0.0 | **Status:** Local Deployment (Phase 1)

An intelligent, configuration-driven AI support system for FixIt home services. Routes 10,000 queries/day across model tiers, reducing LLM costs by ~83% while maintaining >85% customer satisfaction.

---

## 🏗️ Architecture

```
Customer Query
    │
    ▼
REST API (FastAPI)
    │
    ▼
Config Loader ──── config.yaml (AWS Future: AppConfig)
    │
    ▼
Query Router
    ├── Classifier (keyword-based) ─────── (AWS Future: Comprehend)
    ├── Budget Guard (daily limits) ───── (AWS Future: DynamoDB)
    ├── Prompt Manager (versioned YAML) ── (AWS Future: S3 + DynamoDB)
    └── LLM Client (OpenRouter) ────────── (AWS Future: Bedrock)
    │
    ▼
Structured Logger ─── JSONL files ─── (AWS Future: CloudWatch Logs)
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd KDU-2026-AI
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env
# Edit .env:
# - Set OPENROUTER_API_KEY for real LLM calls (free at openrouter.ai/keys)
# - Or keep dry_run_mode: true in config/config.yaml (default, no key needed)
```

### 4. Run Tests

```bash
python -m pytest tests/ -v --tb=short
```

Expected: **≥ 58 tests, all passing**.

### 5. Start the Server

```bash
uvicorn api.main:app --reload --port 8000
```

Visit **http://localhost:8000** for the dashboard.

### 6. Test the API

```bash
# Submit a query (dry-run mode by default)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"My plumber didn't show up and I want a refund\", \"session_id\": \"test-001\"}"

# Check budget status
curl http://localhost:8000/budget/status

# View recent logs
curl http://localhost:8000/logs/recent?n=5

# Hot-reload config (no restart needed)
curl -X POST http://localhost:8000/admin/reload-config
```

---

## 🧠 Model Routing (OpenRouter Free Tier)

All models served via [OpenRouter](https://openrouter.ai) — unified API, all free:

| Tier | Query Type | Model | Size |
|---|---|---|---|
| **Low** | FAQ, simple | `liquid/lfm-2.5-1.2b-instruct:free` | 1.2B |
| **Medium** | Booking, standard | `openai/gpt-oss-20b:free` | 21B MoE |
| **High** | Complaints, complex | `openai/gpt-oss-120b:free` | 120B MoE |
| **Fallback** | Budget limit hit | `liquid/lfm-2.5-1.2b-instruct:free` | 1.2B |

To enable real LLM responses:
1. Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Set `OPENROUTER_API_KEY=sk-or-v1-...` in `.env`
3. Set `dry_run_mode: false` in `config/config.yaml`

---

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Submit a customer query |
| `GET` | `/budget/status` | Current budget state |
| `GET` | `/health` | System health check |
| `POST` | `/admin/reload-config` | Hot-reload config.yaml |
| `GET` | `/admin/prompts` | List prompt versions |
| `GET` | `/admin/config` | View active configuration |
| `GET` | `/logs/recent` | Last N log entries |
| `GET` | `/logs/cost-summary` | Cost analytics |
| `GET` | `/docs` | Swagger interactive UI |

---

## 🗂️ Project Structure

```
KDU-2026-AI/
├── config/
│   ├── config.yaml          # Master configuration (hot-reloadable)
│   └── loader.py            # ConfigLoader singleton
├── router/
│   ├── classifier.py        # Keyword-based query classifier
│   ├── budget_guard.py      # Cost tracking & budget enforcement
│   └── engine.py            # Router orchestration
├── prompts/
│   ├── manager.py           # Prompt loader, renderer, versioner
│   ├── faq/v1.yaml
│   ├── booking/v1.yaml
│   ├── complaint/v1.yaml    # Includes few-shot examples
│   ├── technical/v1.yaml
│   └── fallback/v1.yaml
├── llm/
│   └── client.py            # Unified LLM client (OpenRouter + dry-run)
├── observability/
│   └── logger.py            # Structured JSONL logger
├── api/
│   └── main.py              # FastAPI server
├── tests/                   # 58+ pytest tests
├── logs/                    # Runtime logs (gitignored)
├── DESIGN_PLAN.md           # Full LLMOps design document
└── requirements.txt
```

---

## ☁️ AWS Future Architecture

| Local Component | AWS Replacement | Why |
|---|---|---|
| `config.yaml` | **AWS AppConfig** | Versioned, canary deploys, auto-rollback |
| Budget JSON state | **Amazon DynamoDB** | Atomic increments, multi-instance safe |
| Prompt YAML files | **Amazon S3 + DynamoDB** | Versioned objects, global access |
| Keyword classifier | **Amazon Comprehend** | ML-trained, higher accuracy |
| JSONL log files | **Amazon CloudWatch Logs** | Searchable, alarms, dashboards |
| OpenRouter client | **Amazon Bedrock** | IAM auth, compliance, no API keys |
| FastAPI server | **AWS Lambda + API Gateway** | Serverless, autoscaling, zero idle cost |
| — | **Amazon ElastiCache** | Shared semantic cache across instances |

See `DESIGN_PLAN.md` → Section 10 for the full AWS migration roadmap.

---

## 💰 Cost Analysis

| Scenario | Monthly Cost |
|---|---|
| Before (single model) | $3,000/month |
| After tiered routing | ~$154/month |
| Budget target | $500/month |
| **Savings** | **~83% reduction** |

---

*FixIt AI · LLMOps Design Document v1.0.0*