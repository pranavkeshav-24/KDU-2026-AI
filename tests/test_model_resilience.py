from agents.executor import _is_rate_limited_error
from config import settings
from models.router import get_model_candidates


def test_model_candidates_include_env_fallbacks_without_duplicates(monkeypatch):
    monkeypatch.setattr(settings, "REASONING_MODEL", "model/reasoning")
    monkeypatch.setattr(settings, "FALLBACK_MODEL", "model/fallback")
    monkeypatch.setattr(
        settings,
        "MODEL_FALLBACKS",
        "model/extra-a, model/fallback, model/extra-b",
    )

    assert get_model_candidates("reasoning") == [
        "model/reasoning",
        "model/fallback",
        "model/extra-a",
        "model/extra-b",
    ]


def test_rate_limited_error_detection_matches_common_messages():
    assert _is_rate_limited_error(Exception("Error code: 429"))
    assert _is_rate_limited_error(Exception("temporarily rate-limited upstream"))
    assert _is_rate_limited_error(Exception("Too many requests"))
    assert not _is_rate_limited_error(Exception("No endpoints found"))
