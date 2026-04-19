# router/engine.py
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from config.loader import config_loader
from router.classifier import QueryClassifier, ComplexityTier
from router.budget_guard import BudgetGuard, CostRecord
from prompts.manager import PromptManager
from llm.client import LLMClient
from observability.logger import ObsLogger


@dataclass
class QueryResponse:
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
    budget_utilization_pct: float = 0.0
    prompt_version: str = "v1"
    session_id: Optional[str] = None


class RouterEngine:
    """
    Orchestrates: classify → budget check → prompt load → LLM invoke → log.

    AWS Future Replacement: Amazon Bedrock Agents
    ─────────────────────────────────────────────
    This entire engine can be replaced with Amazon Bedrock Agents, which
    provides managed ReAct orchestration, tool invocation, session memory,
    multi-step reasoning, and built-in guardrails — no custom orchestration
    code to maintain.

    Additionally: AWS Step Functions can orchestrate the classify → route →
    invoke → record workflow with built-in retry, error handling, and
    visual state machine diagrams.

    Migration path:
      1. Define tools as Lambda functions (classifier, budget_check, llm_invoke)
      2. Configure Bedrock Agent with tool definitions + system prompt
      3. Replace RouterEngine.route() with Bedrock Agent invocation
      4. Session memory automatically handled by Bedrock Agent memory
    """

    def __init__(self):
        self.classifier = QueryClassifier()
        self.budget_guard = BudgetGuard()
        self.prompt_manager = PromptManager()
        self.llm_client = LLMClient()
        self.logger = ObsLogger()

    def route(self, query_text: str, session_id: Optional[str] = None) -> QueryResponse:
        query_id = str(uuid.uuid4())[:8]
        config = config_loader.get()
        start_time = datetime.utcnow()
        prompt_version = config.prompts.get("default_version", "v1")

        # Step 1: Classify
        classification = self.classifier.classify(query_text)

        # Step 2: Select model tier
        tier_name = self._complexity_to_tier(classification.complexity)
        model_config = config.models[tier_name]
        fallback_activated = False

        # Step 3: Estimate cost and check budget
        estimated_input_tokens = max(len(query_text.split()) * 2, 50)
        estimated_output_tokens = model_config.max_tokens * 0.5
        estimated_cost = self.budget_guard.estimate_cost(
            model_config.model_id, int(estimated_input_tokens), int(estimated_output_tokens)
        )
        budget_check = self.budget_guard.check_budget(estimated_cost)

        if not budget_check["approved"] and config.features.enable_fallback_on_budget:
            tier_name = "fallback"
            model_config = config.models["fallback"]
            fallback_activated = True
            self.budget_guard.increment_fallback_counter()

        # Step 4: Load prompt
        prompt = self.prompt_manager.render(
            category=classification.category.value,
            version=prompt_version,
            variables={"query": query_text, "service_type": "home_services"},
        )

        # Step 5: Invoke LLM (or dry-run)
        if config.features.dry_run_mode:
            response_text = (
                f"[DRY RUN] Classified as {classification.category.value} "
                f"({classification.complexity.value}) → routed to {tier_name} "
                f"({model_config.model_id}). Query: {query_text[:80]}..."
                if len(query_text) > 80
                else f"[DRY RUN] Classified as {classification.category.value} "
                     f"({classification.complexity.value}) → routed to {tier_name} "
                     f"({model_config.model_id}). Query: {query_text}"
            )
            tokens_in, tokens_out = int(estimated_input_tokens), 60
        else:
            llm_result = self.llm_client.invoke(
                model_id=model_config.model_id,
                provider=model_config.provider,
                prompt=prompt,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
            )
            response_text = llm_result["text"]
            tokens_in = llm_result["tokens_in"]
            tokens_out = llm_result["tokens_out"]

        # Step 6: Record actual cost
        actual_cost = self.budget_guard.estimate_cost(
            model_config.model_id, tokens_in, tokens_out
        )
        self.budget_guard.record_cost(CostRecord(
            timestamp=datetime.utcnow().isoformat(),
            query_id=query_id,
            model_id=model_config.model_id,
            tier=tier_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=actual_cost,
            category=classification.category.value,
        ))

        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        response = QueryResponse(
            query_id=query_id,
            response_text=response_text,
            category=classification.category.value,
            complexity=classification.complexity.value,
            model_used=model_config.model_id,
            tier_used=tier_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=actual_cost,
            latency_ms=round(latency_ms, 2),
            routing_reason=classification.routing_reason,
            fallback_activated=fallback_activated,
            timestamp=datetime.utcnow().isoformat(),
            budget_utilization_pct=round(
                budget_check.get("budget_utilization", 0) * 100, 2
            ),
            prompt_version=prompt_version,
            session_id=session_id,
        )

        # Step 7: Log
        if config.features.enable_structured_logging:
            self.logger.log_request(response)

        return response

    def _complexity_to_tier(self, complexity: ComplexityTier) -> str:
        return {
            ComplexityTier.LOW: "tier_low",
            ComplexityTier.MEDIUM: "tier_medium",
            ComplexityTier.HIGH: "tier_high",
        }[complexity]
