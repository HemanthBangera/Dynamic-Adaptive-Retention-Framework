"""
DARS Layer D – Unit Tests: Schema
===================================
Tests for MemoryPayload, MemoryPoint, DARSWeights, RetentionDecision.
These tests are OFFLINE – they require no network access or Qdrant.

Coverage:
    • Payload construction and default values
    • Serialisation round-trip (to_dict / from_dict)
    • Utility computation formula
    • Weight validation
    • MemoryPoint ID generation
    • RetentionDecision structure
"""

import time
import pytest

from core.layer_d.schema import (
    DARSWeights,
    MemoryPayload,
    MemoryPoint,
    RetentionDecision,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  MemoryPayload
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryPayload:
    """Unit tests for the MemoryPayload dataclass."""

    def test_default_values(self):
        """Fresh payload should have neutral DARS metadata."""
        p = MemoryPayload(text_content="hello world")
        assert p.text_content == "hello world"
        assert p.success_count == 0
        assert p.failure_count == 0
        assert p.utility == 0.0
        assert p.frequency == 0
        assert p.predictive == 0.5
        assert p.is_compressed is False
        assert p.source == ""
        assert p.tags == []
        # Timestamps should be recent
        assert abs(p.recency - time.time()) < 2
        assert abs(p.created_at - time.time()) < 2

    def test_custom_values(self, sample_payload):
        """Payload with custom values should preserve them."""
        p = sample_payload
        assert p.text_content == "The client prefers Python 3.12 for all backend services."
        assert p.success_count == 3
        assert p.failure_count == 1
        assert p.utility == 0.6
        assert p.frequency == 5
        assert p.predictive == 0.7
        assert p.source == "user"
        assert "python" in p.tags

    def test_to_dict(self, sample_payload):
        """to_dict should return a flat dictionary with all fields."""
        d = sample_payload.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "text_content", "success_count", "failure_count", "utility",
            "frequency", "recency", "predictive", "created_at",
            "is_compressed", "source", "tags",
        }
        assert expected_keys.issubset(set(d.keys()))
        assert d["text_content"] == sample_payload.text_content
        assert d["success_count"] == 3

    def test_from_dict_round_trip(self, sample_payload):
        """Serialise → deserialise should produce identical payload."""
        d = sample_payload.to_dict()
        restored = MemoryPayload.from_dict(d)
        assert restored.text_content == sample_payload.text_content
        assert restored.success_count == sample_payload.success_count
        assert restored.failure_count == sample_payload.failure_count
        assert restored.utility == sample_payload.utility
        assert restored.frequency == sample_payload.frequency
        assert restored.predictive == sample_payload.predictive

    def test_from_dict_ignores_extra_keys(self):
        """from_dict should ignore unknown keys gracefully."""
        d = {"text_content": "test", "unknown_field": 42, "another": "xyz"}
        p = MemoryPayload.from_dict(d)
        assert p.text_content == "test"
        assert not hasattr(p, "unknown_field")

    def test_from_dict_handles_missing_keys(self):
        """from_dict with only required field should use defaults."""
        p = MemoryPayload.from_dict({"text_content": "minimal"})
        assert p.text_content == "minimal"
        assert p.success_count == 0
        assert p.predictive == 0.5

    def test_compute_utility_no_history(self):
        """U = 0 / (0 + 0 + 1) = 0.0 for a fresh memory."""
        p = MemoryPayload(text_content="new")
        result = p.compute_utility()
        assert result == 0.0
        assert p.utility == 0.0

    def test_compute_utility_all_success(self):
        """U = 5 / (5 + 0 + 1) = 0.8333..."""
        p = MemoryPayload(text_content="good", success_count=5, failure_count=0)
        result = p.compute_utility()
        assert abs(result - 5 / 6) < 1e-6
        assert p.utility == result

    def test_compute_utility_mixed(self):
        """U = 3 / (3 + 1 + 1) = 0.6"""
        p = MemoryPayload(text_content="mixed", success_count=3, failure_count=1)
        result = p.compute_utility()
        assert abs(result - 0.6) < 1e-6

    def test_compute_utility_all_failure(self):
        """U = 0 / (0 + 5 + 1) = 0.0"""
        p = MemoryPayload(text_content="bad", success_count=0, failure_count=5)
        result = p.compute_utility()
        assert result == 0.0

    def test_compute_utility_updates_in_place(self):
        """compute_utility should mutate the utility field."""
        p = MemoryPayload(text_content="test", success_count=2, failure_count=0)
        assert p.utility == 0.0  # initial default
        p.compute_utility()
        assert abs(p.utility - 2 / 3) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
#  MemoryPoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryPoint:
    """Unit tests for the MemoryPoint dataclass."""

    def test_generate_id_is_unique(self):
        """Each generated ID should be a unique UUID-4 string."""
        ids = {MemoryPoint.generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_id_format(self):
        """Generated ID should match UUID-4 format."""
        pid = MemoryPoint.generate_id()
        import uuid
        parsed = uuid.UUID(pid, version=4)
        assert str(parsed) == pid

    def test_memory_point_construction(self, sample_payload):
        """MemoryPoint should hold vector, payload, and optional scores."""
        mp = MemoryPoint(
            point_id="test-id",
            vector=[0.1] * 384,
            payload=sample_payload,
            score=0.95,
            dars_score=0.72,
        )
        assert mp.point_id == "test-id"
        assert len(mp.vector) == 384
        assert mp.score == 0.95
        assert mp.dars_score == 0.72
        assert mp.payload.text_content == sample_payload.text_content

    def test_memory_point_defaults(self, fresh_payload):
        """Score fields should default to None."""
        mp = MemoryPoint(
            point_id=MemoryPoint.generate_id(),
            vector=[],
            payload=fresh_payload,
        )
        assert mp.score is None
        assert mp.dars_score is None


# ═══════════════════════════════════════════════════════════════════════════════
#  DARSWeights
# ═══════════════════════════════════════════════════════════════════════════════


class TestDARSWeights:
    """Unit tests for the DARS weight vector."""

    def test_default_weights_sum_to_one(self):
        """Default weights (0.3 + 0.2 + 0.3 + 0.2) must equal 1.0."""
        w = DARSWeights()
        assert w.validate() is True
        assert abs(w.w_r + w.w_f + w.w_u + w.w_p - 1.0) < 1e-6

    def test_valid_custom_weights(self):
        """Custom weights summing to 1.0 should validate."""
        w = DARSWeights(w_r=0.25, w_f=0.25, w_u=0.25, w_p=0.25)
        assert w.validate() is True

    def test_invalid_weights_too_high(self):
        """Weights summing > 1.0 should fail validation."""
        w = DARSWeights(w_r=0.5, w_f=0.5, w_u=0.5, w_p=0.5)
        assert w.validate() is False

    def test_invalid_weights_too_low(self):
        """Weights summing < 1.0 should fail validation."""
        w = DARSWeights(w_r=0.1, w_f=0.1, w_u=0.1, w_p=0.1)
        assert w.validate() is False


# ═══════════════════════════════════════════════════════════════════════════════
#  RetentionDecision
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetentionDecision:
    """Unit tests for the RetentionDecision output struct."""

    def test_retain_decision(self):
        d = RetentionDecision(
            action="retain", dars_score=0.85,
            point_id="abc", text_preview="Important memory..."
        )
        assert d.action == "retain"
        assert d.dars_score == 0.85

    def test_compress_decision(self):
        d = RetentionDecision(
            action="compress", dars_score=0.50,
            point_id="def", text_preview="Mid-tier memory..."
        )
        assert d.action == "compress"

    def test_delete_decision(self):
        d = RetentionDecision(
            action="delete", dars_score=0.15,
            point_id="ghi", text_preview="Low value..."
        )
        assert d.action == "delete"
        assert d.dars_score == 0.15
