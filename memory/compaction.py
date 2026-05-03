from __future__ import annotations

import re

from agents_runtime import AgentSpec, SafeRunner
from config import GENERAL_AGENT_MODEL
from memory.case_facts import CaseFacts
from memory.extractor import extract_case_facts
from observability.logger import StructuredLogger


logger = StructuredLogger("memory_compaction")

FILLER_PATTERNS = {"okay", "ok", "cool", "sure", "got it", "thanks", "noted", "alright", "k"}

summarizer_agent = AgentSpec(
    name="Summarizer",
    model=GENERAL_AGENT_MODEL,
    instructions=(
        "Summarize the conversation into 3-5 sentences. Do not include numbers, IDs, amounts, or dates. "
        "Those are preserved separately. Focus on intent, context, and decisions."
    ),
    tools=[],
)


def is_filler(message: str) -> bool:
    stripped = message.strip().lower()
    return stripped in FILLER_PATTERNS or len(stripped) < 8


def redact_structured_values(text: str) -> str:
    text = re.sub(r"\$[\d,]+(?:\.\d{2})?", "[REDACTED_AMOUNT]", text)
    text = re.sub(r"\b\d{9,17}\b", "[REDACTED_NUMBER]", text)
    text = re.sub(r"\bORD-\d+\b", "[REDACTED_ORDER]", text, flags=re.IGNORECASE)
    text = re.sub(PATTERN_DATE, "[REDACTED_DATE]", text)
    return text


PATTERN_DATE = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b"


async def compact_memory(messages: list[str]) -> dict:
    substantive = [message for message in messages if not is_filler(message)]
    if not substantive:
        logger.log("memory_compacted", original_msg_count=len(messages), filtered_count=0, facts_extracted=0)
        return {"case_facts": CaseFacts().__dict__, "summary": "No substantive content.", "requires_user_input": False, "flags": []}

    full_text = "\n".join(substantive)
    facts = extract_case_facts(full_text)
    prose_only = redact_structured_values(full_text)
    summary_result = await SafeRunner.run(summarizer_agent, prose_only)
    logger.log(
        "memory_compacted",
        original_msg_count=len(messages),
        filtered_count=len(substantive),
        facts_extracted=sum(len(v) for k, v in facts.__dict__.items() if isinstance(v, list) and k != "raw_flags"),
    )
    return {
        "case_facts": facts.__dict__,
        "summary": summary_result.final_output,
        "requires_user_input": facts.requires_user_input,
        "flags": facts.raw_flags,
    }

