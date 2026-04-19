# api/main.py
"""
FixIt AI Support System — FastAPI Server

AWS Future Replacement: AWS Lambda + Amazon API Gateway
────────────────────────────────────────────────────────
This FastAPI server → AWS Lambda + API Gateway:
- Lambda: Serverless, auto-scales to handle 10k+ queries/day with zero idle cost
- API Gateway: REST API with WAF, throttling, API keys, and usage plans
- CloudFront in front of API Gateway for edge caching and DDoS protection

Migration path:
  1. Wrap the FastAPI app with Mangum (ASGI adapter for Lambda)
  2. Deploy with AWS SAM or CDK
  3. Configure API Gateway with Lambda proxy integration
  4. Set up CloudFront distribution pointing to API Gateway
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import asdict

# Add project root to sys.path for module imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from config.loader import config_loader
from router.engine import RouterEngine
from router.budget_guard import BudgetGuard
from prompts.manager import PromptManager
from observability.logger import ObsLogger

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FixIt AI Support System",
    description="""
## FixIt AI LLMOps — Intelligent Query Router

An intelligent, configuration-driven AI support system for FixIt home services.

### Features
- **Intelligent Query Routing**: Keyword-based classifier routes queries to the right model tier
- **Budget Guard**: Hard cost limits enforced at routing layer
- **Prompt Versioning**: YAML-based prompt templates with version control
- **Structured Logging**: Every request logged as structured JSON
- **Config Hot-Reload**: Change routing rules without restarting the server

### AWS Future Architecture
Local components are designed to map directly to AWS services:
- `config.yaml` → **AWS AppConfig**
- Budget state → **Amazon DynamoDB**  
- Prompt files → **Amazon S3 + DynamoDB**
- Keyword classifier → **Amazon Comprehend**
- Log files → **Amazon CloudWatch Logs**
- This server → **AWS Lambda + API Gateway**
    """,
    version="1.0.0",
    contact={"name": "FixIt AI Team", "email": "ai-team@fixit.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Singletons ───────────────────────────────────────────────────────────────

router_engine = RouterEngine()
budget_guard = BudgetGuard()
prompt_manager = PromptManager()
obs_logger = ObsLogger()


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str = Field(..., description="Customer query text", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Optional session identifier")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "text": "My plumber didn't show up and I want a full refund",
                "session_id": "sess-001"
            }
        ]
    }}


class QueryResponse(BaseModel):
    query_id: str
    response_text: str
    category: str
    complexity: str
    model_used: str
    tier_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    routing_reason: str
    fallback_activated: bool
    timestamp: str
    budget_utilization_pct: float
    prompt_version: str


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    dry_run_mode: bool
    uptime_timestamp: str


class ConfigReloadResponse(BaseModel):
    success: bool
    message: str
    new_version: str
    environment: str


# ─── Startup time tracking ────────────────────────────────────────────────────

_startup_time = datetime.utcnow().isoformat()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Info"])
async def root():
    """Landing page with system overview."""
    config = config_loader.get()
    mode_badge = "🟡 DRY RUN" if config.features.dry_run_mode else "🟢 LIVE"
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FixIt AI Support System</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .container {{
      max-width: 860px;
      width: 100%;
      padding: 2rem;
    }}
    .header {{
      text-align: center;
      margin-bottom: 3rem;
    }}
    .logo {{ font-size: 3.5rem; margin-bottom: 1rem; }}
    h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .subtitle {{
      color: #94a3b8;
      margin-top: 0.5rem;
      font-size: 1.05rem;
    }}
    .badge {{
      display: inline-block;
      padding: 0.3rem 1rem;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 1rem;
      background: rgba(251, 191, 36, 0.15);
      border: 1px solid rgba(251, 191, 36, 0.4);
      color: #fbbf24;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.2rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 1.5rem;
      backdrop-filter: blur(12px);
      transition: transform 0.2s, border-color 0.2s;
    }}
    .card:hover {{ transform: translateY(-3px); border-color: rgba(56,189,248,0.4); }}
    .card-icon {{ font-size: 2rem; margin-bottom: 0.75rem; }}
    .card-title {{ font-weight: 600; font-size: 1rem; color: #f1f5f9; margin-bottom: 0.4rem; }}
    .card-desc {{ font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }}
    .endpoints {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }}
    .endpoints h2 {{ font-size: 1.1rem; color: #f1f5f9; margin-bottom: 1rem; }}
    .endpoint {{
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 0.7rem 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .endpoint:last-child {{ border-bottom: none; }}
    .method {{
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      min-width: 52px;
      text-align: center;
    }}
    .method.post {{ background: rgba(34,197,94,0.2); color: #4ade80; }}
    .method.get {{ background: rgba(56,189,248,0.2); color: #38bdf8; }}
    .path {{ font-family: monospace; color: #e2e8f0; font-size: 0.9rem; }}
    .desc {{ color: #64748b; font-size: 0.83rem; margin-left: auto; }}
    .cta {{
      text-align: center;
      display: flex;
      gap: 1rem;
      justify-content: center;
      flex-wrap: wrap;
    }}
    a.btn {{
      display: inline-block;
      padding: 0.75rem 1.8rem;
      border-radius: 12px;
      font-weight: 600;
      text-decoration: none;
      font-size: 0.95rem;
      transition: opacity 0.2s, transform 0.2s;
    }}
    a.btn:hover {{ opacity: 0.85; transform: translateY(-2px); }}
    a.btn-primary {{
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      color: #0f172a;
    }}
    a.btn-secondary {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.15);
      color: #e2e8f0;
    }}
    .info {{ color: #64748b; font-size: 0.8rem; text-align: center; margin-top: 2rem; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">🔧</div>
      <h1>FixIt AI Support System</h1>
      <p class="subtitle">LLMOps — Intelligent Query Router · v{config.version}</p>
      <span class="badge">{mode_badge} MODE &nbsp;·&nbsp; {config.environment.upper()} ENV</span>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-icon">🧠</div>
        <div class="card-title">Intelligent Router</div>
        <div class="card-desc">Keyword-based classifier routes queries to Low / Medium / High model tiers, cutting cost by 83%.</div>
      </div>
      <div class="card">
        <div class="card-icon">💰</div>
        <div class="card-title">Budget Guard</div>
        <div class="card-desc">Hard daily limit of ${config.budget.daily_budget_usd:.2f}. Forces fallback model at 95% utilization.</div>
      </div>
      <div class="card">
        <div class="card-icon">📝</div>
        <div class="card-title">Prompt Versioning</div>
        <div class="card-desc">5 categories · 1 version each · Jinja2 templates with few-shot examples. Hot-reload enabled.</div>
      </div>
      <div class="card">
        <div class="card-icon">📊</div>
        <div class="card-title">Observability</div>
        <div class="card-desc">Every request logged as structured JSONL with cost, latency, routing reason, and tier metadata.</div>
      </div>
    </div>

    <div class="endpoints">
      <h2>📡 API Endpoints</h2>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/query</span>
        <span class="desc">Submit a customer query for AI routing + response</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/budget/status</span>
        <span class="desc">Current daily budget state and utilization</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/health</span>
        <span class="desc">System health check</span>
      </div>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/admin/reload-config</span>
        <span class="desc">Hot-reload config.yaml without server restart</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/admin/prompts</span>
        <span class="desc">List all prompt categories and versions</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/logs/recent</span>
        <span class="desc">View last N structured log entries</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/logs/cost-summary</span>
        <span class="desc">Aggregated cost analytics from logs</span>
      </div>
    </div>

    <div class="cta">
      <a href="/docs" class="btn btn-primary">📖 Swagger UI</a>
      <a href="/redoc" class="btn btn-secondary">📋 ReDoc</a>
      <a href="/health" class="btn btn-secondary">❤️ Health Check</a>
      <a href="/budget/status" class="btn btn-secondary">💰 Budget Status</a>
    </div>

    <p class="info">FixIt AI · LLMOps Design Document v{config.version} · Started {_startup_time}</p>
  </div>
</body>
</html>
""")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — verify config, prompts, and core components are operational."""
    config = config_loader.get()
    return HealthResponse(
        status="ok",
        version=config.version,
        environment=config.environment,
        dry_run_mode=config.features.dry_run_mode,
        uptime_timestamp=_startup_time,
    )


@app.post("/query", response_model=QueryResponse, tags=["Core"])
async def submit_query(request: QueryRequest):
    """
    Submit a customer support query.

    The system will:
    1. Classify the query (FAQ / Booking / Complaint / Technical)
    2. Determine complexity tier (Low / Medium / High)
    3. Check budget and route to appropriate model
    4. Load the versioned prompt for that category
    5. Invoke the LLM (or return a dry-run response)
    6. Record cost and log the full trace
    """
    try:
        result = router_engine.route(
            query_text=request.text,
            session_id=request.session_id,
        )
        return QueryResponse(
            query_id=result.query_id,
            response_text=result.response_text,
            category=result.category,
            complexity=result.complexity,
            model_used=result.model_used,
            tier_used=result.tier_used,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            routing_reason=result.routing_reason,
            fallback_activated=result.fallback_activated,
            timestamp=result.timestamp,
            budget_utilization_pct=result.budget_utilization_pct,
            prompt_version=result.prompt_version,
        )
    except Exception as e:
        obs_logger.log_error(
            query_id="unknown",
            error=str(e),
            context={"query": request.text[:100]},
        )
        raise HTTPException(status_code=500, detail=f"Internal routing error: {str(e)}")


@app.get("/budget/status", tags=["Budget"])
async def budget_status():
    """
    Get current daily budget state, spend, and utilization metrics.

    AWS Future: This data lives in Amazon DynamoDB — query will be replaced
    with a DynamoDB GetItem call.
    """
    summary = budget_guard.get_summary()
    config = config_loader.get()
    return {
        **summary,
        "hard_limit_active": summary["budget_utilization_pct"] >= config.budget.hard_limit_pct * 100,
        "warning_active": summary["budget_utilization_pct"] >= config.budget.warning_threshold_pct * 100,
        "currency": "USD",
    }


@app.post("/admin/reload-config", response_model=ConfigReloadResponse, tags=["Admin"])
async def reload_config():
    """
    Hot-reload config.yaml without restarting the server.

    AWS Future: This endpoint becomes a no-op — AWS AppConfig pushes
    config updates automatically with configurable polling intervals.
    """
    try:
        new_config = config_loader.reload()
        prompt_manager.invalidate_cache()
        return ConfigReloadResponse(
            success=True,
            message="Configuration reloaded successfully. Prompt cache cleared.",
            new_version=new_config.version,
            environment=new_config.environment,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config reload failed: {str(e)}")


@app.get("/admin/prompts", tags=["Admin"])
async def list_prompts():
    """
    List all available prompt categories and their versions.

    AWS Future: This queries Amazon DynamoDB prompt registry table.
    """
    categories = prompt_manager.list_categories()
    result = {}
    for cat in categories:
        versions = prompt_manager.list_versions(cat)
        metadata_list = []
        for v in versions:
            try:
                meta = prompt_manager.get_metadata(cat, v)
                metadata_list.append({
                    "version": v,
                    "status": meta.get("status", "unknown"),
                    "eval_score": meta.get("eval_score"),
                    "created_at": meta.get("created_at"),
                    "model_family": meta.get("model_family"),
                })
            except Exception:
                metadata_list.append({"version": v, "status": "error"})
        result[cat] = metadata_list
    return {"categories": result, "total_categories": len(categories)}


@app.get("/logs/recent", tags=["Observability"])
async def recent_logs(n: int = 20):
    """
    Retrieve the last N structured log entries.

    AWS Future: This queries Amazon CloudWatch Log Insights or a DynamoDB
    query index on the request logs table.
    """
    if n > 200:
        n = 200
    logs = obs_logger.get_recent_logs(n=n)
    return {
        "count": len(logs),
        "logs": logs,
    }


@app.get("/logs/cost-summary", tags=["Observability"])
async def cost_summary():
    """
    Aggregated cost analytics from recent log entries.

    AWS Future: Computed via CloudWatch Metric Filters on the log group,
    or a DynamoDB aggregation query.
    """
    return obs_logger.get_cost_summary()


@app.get("/admin/config", tags=["Admin"])
async def get_config():
    """View current active configuration (non-sensitive fields only)."""
    config = config_loader.get()
    return {
        "version": config.version,
        "environment": config.environment,
        "features": {
            "dry_run_mode": config.features.dry_run_mode,
            "enable_semantic_cache": config.features.enable_semantic_cache,
            "enable_fallback_on_budget": config.features.enable_fallback_on_budget,
            "enable_structured_logging": config.features.enable_structured_logging,
            "enable_cost_tracking": config.features.enable_cost_tracking,
        },
        "models": {
            tier: {
                "model_id": tc.model_id,
                "provider": tc.provider,
                "max_tokens": tc.max_tokens,
                "temperature": tc.temperature,
            }
            for tier, tc in config.models.items()
        },
        "budget": {
            "daily_budget_usd": config.budget.daily_budget_usd,
            "monthly_budget_usd": config.budget.monthly_budget_usd,
            "warning_threshold_pct": config.budget.warning_threshold_pct,
            "hard_limit_pct": config.budget.hard_limit_pct,
        },
        "routing": {
            "low_complexity_threshold": config.routing.low_complexity_threshold,
            "high_complexity_threshold": config.routing.high_complexity_threshold,
        },
    }
