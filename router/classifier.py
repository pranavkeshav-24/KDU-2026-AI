# router/classifier.py
import re
from dataclasses import dataclass
from enum import Enum
from typing import List
from config.loader import config_loader


class QueryCategory(Enum):
    FAQ = "faq"
    BOOKING = "booking"
    COMPLAINT = "complaint"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class ComplexityTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ClassificationResult:
    category: QueryCategory
    complexity: ComplexityTier
    confidence: float
    matched_keywords: List[str]
    routing_reason: str


class QueryClassifier:
    """
    Rule-based classifier using keyword scoring.

    Design Decision: Rule-based chosen over ML classifier for:
    1. Zero inference cost (runs locally, no model invocation)
    2. Fully debuggable and auditable
    3. Config-driven — update keywords in config.yaml, no code change
    4. Deterministic — same input always produces same routing decision

    AWS Future Replacement: Amazon Comprehend (Custom Classifier)
    ─────────────────────────────────────────────────────────────
    Amazon Comprehend trains a custom document classifier on labeled
    FixIt query data. This provides:
    - Higher accuracy on edge cases and mixed-intent queries
    - Active learning: new misclassified queries feed back into training
    - No keyword maintenance as vocabulary evolves
    - Real-time endpoint via Comprehend API (sub-50ms)

    Migration path:
      1. Export historical query logs with category labels
      2. Train Comprehend custom classifier on ≥ 50 examples/category
      3. Deploy as Comprehend endpoint
      4. Replace this class with a ComprehendClassifier wrapper
    """

    def classify(self, query_text: str) -> ClassificationResult:
        config = config_loader.get()
        routing = config.routing
        query_lower = query_text.lower().strip() if query_text else ""

        matched_keywords: List[str] = []
        scores = {
            "faq": 0.0,
            "booking": 0.0,
            "complaint": 0.0,
            "high_complexity": 0.0,
        }

        # Score against each keyword list
        # high_complexity keywords: 0.75 per hit — single hit guarantees HIGH tier (> 0.7 threshold)
        for kw in routing.high_complexity_keywords:
            if kw.lower() in query_lower:
                scores["high_complexity"] += 0.75
                matched_keywords.append(f"high:{kw}")
                break  # One hit is enough to confirm high complexity; stop over-counting

        for kw in routing.complaint_keywords:
            if kw.lower() in query_lower:
                scores["complaint"] += 0.35
                matched_keywords.append(f"complaint:{kw}")

        for kw in routing.booking_keywords:
            if kw.lower() in query_lower:
                scores["booking"] += 0.25
                matched_keywords.append(f"booking:{kw}")

        for kw in routing.faq_keywords:
            if kw.lower() in query_lower:
                scores["faq"] += 0.15
                matched_keywords.append(f"faq:{kw}")

        # Query length heuristic (long queries tend to be more complex)
        word_count = len(query_text.split()) if query_text else 0
        length_score = min(word_count / 30.0, 0.3)

        # Determine category and complexity score
        if scores["complaint"] > 0 or scores["high_complexity"] > 0:
            category = QueryCategory.COMPLAINT
            complexity_score = min(
                scores["complaint"] + scores["high_complexity"] + length_score, 1.0
            )
        elif scores["booking"] > 0:
            category = QueryCategory.BOOKING
            # Cap booking at 0.65 so multiple booking keywords never cross HIGH threshold
            # Only high_complexity keywords (refund, damage, etc.) should push to HIGH
            raw_booking_score = min(scores["booking"] + length_score, 1.0)
            complexity_score = min(raw_booking_score, 0.65)
        elif scores["faq"] > 0:
            category = QueryCategory.FAQ
            complexity_score = min(scores["faq"] + length_score * 0.5, 1.0)
        else:
            category = QueryCategory.UNKNOWN
            complexity_score = 0.3 + length_score  # default to medium for unknowns

        # Map score → complexity tier
        if complexity_score < routing.low_complexity_threshold:
            complexity = ComplexityTier.LOW
        elif complexity_score > routing.high_complexity_threshold:
            complexity = ComplexityTier.HIGH
        else:
            complexity = ComplexityTier.MEDIUM

        return ClassificationResult(
            category=category,
            complexity=complexity,
            confidence=round(complexity_score, 3),
            matched_keywords=matched_keywords,
            routing_reason=(
                f"score={complexity_score:.2f}, "
                f"category={category.value}, "
                f"keywords={matched_keywords[:3]}"
            ),
        )
