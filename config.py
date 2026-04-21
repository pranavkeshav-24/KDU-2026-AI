from dataclasses import dataclass
from typing import Dict

# Summarization model switched to BART as requested.
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
REFINER_MODEL = "sshleifer/distilbart-cnn-12-6"
QA_MODEL = "deepset/roberta-base-squad2"

# Keep headroom under BART's practical context window.
CHUNK_MAX_TOKENS = 900

# If merged summaries exceed this budget, run one extra reduction pass.
MERGED_REDUCTION_TOKEN_THRESHOLD = 500

LENGTH_CONFIG: Dict[str, Dict[str, int]] = {
    "short": {"min_length": 35, "max_length": 90},
    "medium": {"min_length": 90, "max_length": 200},
    "long": {"min_length": 180, "max_length": 420},
}

VALID_LENGTHS = set(LENGTH_CONFIG.keys())

QA_CONFIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class AppTheme:
    title: str = "Tri-Model AI Assistant"
    subtitle: str = "BART Summarization + DistilBART Refinement + RoBERTa QA"
