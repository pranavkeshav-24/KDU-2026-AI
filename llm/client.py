# llm/client.py
import os
import time
import random
from typing import Dict, Any


class LLMClient:
    """
    Unified LLM client interface.

    Primary provider: OpenRouter (https://openrouter.ai)
    OpenRouter exposes an OpenAI-compatible API for 100+ models.
    All selected models are free-tier via OpenRouter.

    Model tier mapping:
    - tier_low    → liquid/lfm-2.5-1.2b-instruct:free  (1.2B, fastest)
    - tier_medium → openai/gpt-oss-20b:free             (21B MoE)
    - tier_high   → openai/gpt-oss-120b:free            (120B MoE, full reasoning)
    - fallback    → liquid/lfm-2.5-1.2b-instruct:free   (same as tier_low)

    Implements retry logic with exponential backoff for transient failures.

    AWS Future Replacement: Amazon Bedrock
    ────────────────────────────────────────
    Replace OpenRouter with Amazon Bedrock's unified model API:
    - Single boto3 client for all models (Claude, Mistral, Llama, etc.)
    - IAM-based auth — no API keys to rotate or store
    - Built-in compliance (HIPAA, SOC2) and data residency guarantees
    - Bedrock Guardrails for content filtering at the infrastructure level
    - Model version pinning and deprecation management

    Migration path:
      1. Replace openai.OpenAI(base_url=openrouter) with
         boto3.client('bedrock-runtime', region_name='us-east-1')
      2. Update model IDs to Bedrock format (e.g. 'amazon.titan-text-express-v1')
      3. Use invoke_model() with the same prompt structure
      4. Remove API key management — use IAM role instead
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Realistic customer support responses used in dry-run mode
    _DRY_RUN_RESPONSES = {
        "faq": [
            "FixIt operates Monday–Saturday 7am–9pm and Sunday 9am–6pm. "
            "You can book online at fixit.com, via our app, or call 1-800-FIXIT-NOW!",
            "Our services start at $75 for a standard call-out. You'll receive a "
            "fixed quote before any work begins — no hidden fees guaranteed.",
            "Yes! We offer plumbing, electrical, and cleaning services across all "
            "major metro areas. Same-day and next-day slots are often available.",
        ],
        "booking": [
            "Done — I've submitted your reschedule request. You'll receive a confirmation "
            "with your new time slot within 30 minutes during business hours.",
            "Your cancellation is confirmed. Since it's within the free window (4+ hours), "
            "you'll receive a full refund within 3–5 business days.",
            "I've checked availability and we have slots open on Thursday and Friday. "
            "Which date works better for you?",
        ],
        "complaint": [
            "I'm truly sorry this happened — this is not the standard FixIt holds itself to. "
            "I've flagged your case as priority and I'm processing a full refund right now "
            "(3–5 business days). I'd also like to offer a 20% discount on your next booking.",
            "I completely understand your frustration, and I sincerely apologize on behalf "
            "of FixIt. I'm arranging a priority re-service at zero cost within 48 hours. "
            "Would tomorrow or the day after work for you?",
        ],
        "technical": [
            "That error code typically indicates a sensor fault. Steps: 1) Turn the unit "
            "off and wait 30 mins, 2) Press the reset button, 3) Power back on. If the "
            "error persists, I'd recommend booking a FixIt specialist for a safe inspection.",
            "Based on your description, this sounds like a pressure issue in your water line. "
            "First check if your main shutoff valve is fully open. If the problem continues, "
            "please book a plumber — further DIY may be unsafe.",
        ],
        "fallback": [
            "Thank you for contacting FixIt support! I'm here to help. Could you share a "
            "bit more detail so I can assist you better? You can also call "
            "1-800-FIXIT-NOW for immediate assistance.",
        ],
        "unknown": [
            "Thank you for reaching out! I want to make sure I help you correctly. Could "
            "you clarify what you need assistance with? Visit fixit.com/support for all options.",
        ],
    }

    def invoke(
        self,
        model_id: str,
        provider: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Invoke the LLM with retry logic and exponential backoff.

        Returns dict with: text, tokens_in, tokens_out, model_id, provider
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if provider == "openrouter":
                    return self._openrouter_invoke(model_id, prompt, temperature, max_tokens)
                elif provider == "dry_run":
                    return self._dry_run_invoke(model_id, prompt)
                else:
                    # Unknown provider — safe fallback to dry run
                    return self._dry_run_invoke(model_id, prompt)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s
                    wait_time = (2 ** attempt) + random.uniform(0, 0.3)
                    time.sleep(wait_time)
                    continue

        # All retries exhausted — return graceful degradation response
        return {
            "text": (
                "I'm experiencing technical difficulties right now. "
                "Please try again in a moment or call 1-800-FIXIT-NOW for immediate help. "
                f"(ref: {str(last_error)[:80]})"
            ),
            "tokens_in": 50,
            "tokens_out": 30,
            "model_id": model_id,
            "provider": provider,
            "error": str(last_error),
        }

    def _openrouter_invoke(
        self, model_id: str, prompt: str, temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        """
        Calls OpenRouter API using the OpenAI-compatible interface.
        Requires OPENROUTER_API_KEY environment variable.

        OpenRouter docs: https://openrouter.ai/docs
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your free key at https://openrouter.ai/keys. "
                "Or set dry_run_mode: true in config.yaml to skip real LLM calls."
            )

        client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=api_key,
        )

        system_text, user_text = self._parse_prompt(prompt)

        extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "FixIt-AI-Local"),
        }

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=extra_headers,
        )

        response_text = completion.choices[0].message.content or ""
        usage = completion.usage

        return {
            "text": response_text,
            "tokens_in": usage.prompt_tokens if usage else int(len(prompt.split()) * 1.3),
            "tokens_out": usage.completion_tokens if usage else len(response_text.split()),
            "model_id": model_id,
            "provider": "openrouter",
        }

    def _dry_run_invoke(self, model_id: str, prompt: str) -> Dict[str, Any]:
        """Returns a realistic simulated response without calling any API."""
        prompt_lower = prompt.lower()

        # Detect category from prompt content
        category = "unknown"
        if "complaint" in prompt_lower or "empathetic" in prompt_lower or "senior customer support" in prompt_lower:
            category = "complaint"
        elif "booking" in prompt_lower or "reschedule" in prompt_lower or "cancel" in prompt_lower:
            category = "booking"
        elif "technical" in prompt_lower or "error code" in prompt_lower or "diagnostic" in prompt_lower:
            category = "technical"
        elif "faq" in prompt_lower or "information assistant" in prompt_lower:
            category = "faq"
        elif "fallback" in prompt_lower or len(prompt) < 200:
            category = "fallback"

        responses = self._DRY_RUN_RESPONSES.get(category, self._DRY_RUN_RESPONSES["unknown"])
        response_text = random.choice(responses)

        # Realistic token estimation
        tokens_in = max(int(len(prompt.split()) * 1.3), 50)
        tokens_out = max(int(len(response_text.split()) * 1.3), 20)

        return {
            "text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model_id": model_id,
            "provider": "dry_run",
        }

    def _parse_prompt(self, prompt: str):
        """Parse system and user parts from the [SYSTEM]/[USER] formatted prompt string."""
        system_text = ""
        user_text = prompt

        if "[SYSTEM]" in prompt and "[USER]" in prompt:
            parts = prompt.split("[USER]")
            system_text = parts[0].replace("[SYSTEM]", "").strip()
            user_text = parts[1].strip() if len(parts) > 1 else ""

        return system_text, user_text
