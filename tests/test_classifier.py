# tests/test_classifier.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from router.classifier import QueryClassifier, QueryCategory, ComplexityTier


class TestQueryClassifier:

    def setup_method(self):
        self.classifier = QueryClassifier()

    # ── Category Classification ──────────────────────────────────────

    def test_faq_low_complexity(self):
        result = self.classifier.classify("What are your hours?")
        assert result.category == QueryCategory.FAQ
        assert result.complexity == ComplexityTier.LOW
        assert result.confidence < 0.5

    def test_faq_price_query(self):
        result = self.classifier.classify("How much does a plumbing service cost?")
        assert result.category == QueryCategory.FAQ

    def test_complaint_high_complexity(self):
        result = self.classifier.classify(
            "My plumber didn't show up and I want a full refund immediately"
        )
        assert result.category == QueryCategory.COMPLAINT
        assert result.complexity == ComplexityTier.HIGH
        assert result.confidence > 0.7

    def test_booking_medium_complexity(self):
        result = self.classifier.classify(
            "Can I reschedule my cleaning appointment to next Thursday?"
        )
        assert result.category == QueryCategory.BOOKING
        # Booking is capped at 0.65, so it can be LOW or MEDIUM but not HIGH
        # (unless it also contains high_complexity keywords like 'refund')
        assert result.complexity in [ComplexityTier.LOW, ComplexityTier.MEDIUM]
        assert result.confidence <= 0.65

    def test_booking_cancel(self):
        result = self.classifier.classify("I need to cancel my appointment for tomorrow")
        assert result.category == QueryCategory.BOOKING

    def test_multiple_keywords_escalates_complexity(self):
        result = self.classifier.classify(
            "I am angry, the electrician damaged my wall and I need a refund"
        )
        assert result.complexity == ComplexityTier.HIGH
        assert len(result.matched_keywords) >= 2

    def test_complaint_scam_keyword(self):
        result = self.classifier.classify("This service is a scam, worst experience ever")
        assert result.category == QueryCategory.COMPLAINT
        assert result.complexity in [ComplexityTier.MEDIUM, ComplexityTier.HIGH]

    def test_complaint_emergency_keyword(self):
        # "emergency" is in high_complexity_keywords — single hit = 0.75 > 0.7 threshold
        result = self.classifier.classify("I have a plumbing emergency at my house!")
        assert result.complexity == ComplexityTier.HIGH
        assert result.confidence >= 0.75

    # ── Edge Cases ────────────────────────────────────────────────────

    def test_empty_query_handled(self):
        result = self.classifier.classify("")
        assert result.category is not None  # Should not raise

    def test_very_long_query_escalates(self):
        long_query = " ".join(["word"] * 50)  # 50 words → length heuristic
        result = self.classifier.classify(long_query)
        # Length alone should push above LOW
        assert result.complexity in [ComplexityTier.MEDIUM, ComplexityTier.HIGH]

    def test_special_characters_handled(self):
        result = self.classifier.classify("What's the price??? 😡 !!")
        assert result is not None  # Should not raise

    def test_unknown_category_defaults_to_medium(self):
        result = self.classifier.classify("xyzzy nonsense words qwerty")
        assert result.category == QueryCategory.UNKNOWN
        # Unknown should default to medium range (0.3 + length)
        assert result.complexity in [ComplexityTier.LOW, ComplexityTier.MEDIUM]

    # ── Routing Reason / Metadata ─────────────────────────────────────

    def test_routing_reason_populated(self):
        result = self.classifier.classify("I want a refund")
        assert len(result.routing_reason) > 0
        assert "score=" in result.routing_reason

    def test_matched_keywords_populated(self):
        result = self.classifier.classify("I want a refund for this complaint")
        assert len(result.matched_keywords) >= 1

    def test_confidence_is_float_in_range(self):
        result = self.classifier.classify("Can I book an appointment?")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_faq_low_confidence_threshold(self):
        """Short FAQ queries should score below the high threshold."""
        result = self.classifier.classify("What is your service area?")
        assert result.confidence < 0.9  # not everything is high

    def test_complaint_high_confidence(self):
        result = self.classifier.classify("My plumber never arrived, I want a refund!")
        assert result.confidence > 0.5

    # ── Complexity Tier Mapping ───────────────────────────────────────

    def test_faq_keyword_produces_low_or_medium(self):
        result = self.classifier.classify("Do you provide cleaning services?")
        assert result.complexity in [ComplexityTier.LOW, ComplexityTier.MEDIUM]

    def test_high_complexity_keyword_alone_is_enough(self):
        # A single high_complexity keyword scores 0.75 > threshold of 0.7 → HIGH
        result = self.classifier.classify("I need a refund")
        assert result.complexity == ComplexityTier.HIGH
        assert result.confidence >= 0.75

    def test_overcharge_escalates_to_high(self):
        # "overcharge" is a high_complexity keyword — single hit = 0.75 > 0.7
        result = self.classifier.classify("I was overcharged on my invoice")
        assert result.complexity == ComplexityTier.HIGH
        assert result.confidence >= 0.75
