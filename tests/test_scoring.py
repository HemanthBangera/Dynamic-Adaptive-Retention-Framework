"""
DARS Layer D – Unit Tests: DARS Scoring Engine
================================================
Tests for the DARS score computation, retention classification,
and the mathematical formulas (recency decay, frequency normalisation,
utility credit assignment).

These tests instantiate MemoryVault but mock the Qdrant client
where needed to avoid network dependency.

Coverage:
    • Recency exponential decay  R = e^(−λΔt)
    • Frequency log-normalisation  F = log(1+f) / log(1+cap)
    • Utility formula  U = success / (success + failure + 1)
    • Full DARS score composition  S = w_r·R + w_f·F + w_u·U + w_p·P
    • Retention classification thresholds
    • Edge cases (zero scores, max scores, boundary values)
"""

import math
import time
import pytest

from config.settings import DARSConfig
from core.layer_d.schema import MemoryPayload, DARSWeights
from core.layer_d.storage import MemoryVault


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    return DARSConfig()


@pytest.fixture
def vault(config, monkeypatch):
    """
    Create a MemoryVault that SKIPS Qdrant connection.
    We only need the scoring / classification methods.
    """
    # Monkeypatch QdrantClient to avoid network call
    import core.layer_d.storage as storage_module
    from unittest.mock import MagicMock

    original_init = MemoryVault.__init__

    def patched_init(self, config=None, collection_name=None):
        self.config = config or DARSConfig()
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        self.client = MagicMock()  # Mock Qdrant client
        self.embedder = MagicMock()  # Mock embedder
        self.weights = DARSWeights(
            w_r=self.config.WEIGHT_RECENCY,
            w_f=self.config.WEIGHT_FREQUENCY,
            w_u=self.config.WEIGHT_UTILITY,
            w_p=self.config.WEIGHT_PREDICTIVE,
        )

    monkeypatch.setattr(MemoryVault, "__init__", patched_init)
    v = MemoryVault(config=config, collection_name="test_scoring")
    monkeypatch.setattr(MemoryVault, "__init__", original_init)
    return v


# ═══════════════════════════════════════════════════════════════════════════════
#  Recency Tests  –  R = e^(−λ·Δt_hours)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecency:
    """Tests for the exponential recency decay function."""

    def test_zero_delay(self, vault):
        """Just-accessed memory should have R ≈ 1.0."""
        now = time.time()
        R = vault._compute_recency(now, current_time=now)
        assert abs(R - 1.0) < 1e-6

    def test_one_hour_delay(self, vault, config):
        """After 1 hour, R = e^(−λ·1)."""
        now = time.time()
        one_hour_ago = now - 3600
        R = vault._compute_recency(one_hour_ago, current_time=now)
        expected = math.exp(-config.RECENCY_DECAY_LAMBDA * 1.0)
        assert abs(R - expected) < 1e-6

    def test_24_hour_delay(self, vault, config):
        """After 24 hours, recency should be noticeably decayed."""
        now = time.time()
        yesterday = now - 86400
        R = vault._compute_recency(yesterday, current_time=now)
        expected = math.exp(-config.RECENCY_DECAY_LAMBDA * 24.0)
        assert abs(R - expected) < 1e-6
        assert R < 1.0

    def test_very_old_memory(self, vault):
        """After 30 days, recency should be very low."""
        now = time.time()
        month_ago = now - (30 * 86400)
        R = vault._compute_recency(month_ago, current_time=now)
        assert R < 0.5, f"30-day-old memory should have low recency, got {R}"

    def test_future_timestamp(self, vault):
        """If last_access is in the future (edge case), R should be 1.0."""
        now = time.time()
        future = now + 3600
        R = vault._compute_recency(future, current_time=now)
        # max(0, delta) ensures no negative time → R = e^0 = 1.0
        assert abs(R - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
#  Frequency Tests  –  F = log(1+f) / log(1+cap)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFrequency:
    """Tests for the log-normalised frequency function."""

    def test_zero_access(self, vault):
        """Never-accessed memory → F = log(1)/log(51) = 0.0."""
        F = vault._compute_frequency(0)
        assert F == 0.0

    def test_one_access(self, vault, config):
        """F = log(2) / log(51) ≈ 0.176."""
        F = vault._compute_frequency(1)
        expected = math.log(2) / math.log(1 + config.FREQUENCY_CAP)
        assert abs(F - expected) < 1e-6

    def test_cap_access(self, vault, config):
        """At the cap, F should be 1.0."""
        F = vault._compute_frequency(config.FREQUENCY_CAP)
        assert abs(F - 1.0) < 1e-6

    def test_above_cap(self, vault, config):
        """Above the cap, F should be clamped at 1.0."""
        F = vault._compute_frequency(config.FREQUENCY_CAP * 2)
        assert abs(F - 1.0) < 1e-6

    def test_monotonic_increase(self, vault):
        """Frequency score should increase with access count."""
        scores = [vault._compute_frequency(i) for i in range(0, 51, 10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility Tests  –  U = success / (success + failure + 1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUtility:
    """Tests for the utility credit assignment formula."""

    def test_no_history(self, vault):
        """U = 1 / (0 + 0 + 2) = 0.5."""
        U = vault._compute_utility_score(0, 0)
        assert U == 0.5

    def test_perfect_success(self, vault):
        """U = 11 / (10 + 0 + 2) ≈ 0.916."""
        U = vault._compute_utility_score(10, 0)
        assert abs(U - 11 / 12) < 1e-6

    def test_equal_success_failure(self, vault):
        """U = 6 / (5 + 5 + 2) = 0.5."""
        U = vault._compute_utility_score(5, 5)
        assert abs(U - 6 / 12) < 1e-6

    def test_all_failure(self, vault):
        """U = 1 / (0 + 10 + 2) ≈ 0.083."""
        U = vault._compute_utility_score(0, 10)
        assert abs(U - 1 / 12) < 1e-6

    def test_utility_range(self, vault):
        """Utility should always be in [0, 1)."""
        for s in range(0, 100):
            for f in range(0, 10):
                U = vault._compute_utility_score(s, f)
                assert 0.0 <= U < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Full DARS Score  –  S = w_r·R + w_f·F + w_u·U + w_p·P
# ═══════════════════════════════════════════════════════════════════════════════


class TestDARSScore:
    """Tests for the composite DARS scoring function."""

    def test_fresh_memory(self, vault):
        """Brand-new memory: R≈1, F=0, U=0.5, P=0.5 → known value."""
        now = time.time()
        payload = {
            "recency": now,
            "frequency": 0,
            "success_count": 0,
            "failure_count": 0,
            "predictive": 0.5,
        }
        S = vault.compute_dars_score(payload, current_time=now)
        expected_s = vault.weights.w_r * 1.0 + vault.weights.w_f * 0.0 + vault.weights.w_u * 0.5 + vault.weights.w_p * 0.5
        assert abs(S - expected_s) < 1e-4

    def test_perfect_memory(self, vault):
        """Maximum scores across all dimensions → S should be high."""
        now = time.time()
        payload = {
            "recency": now,
            "frequency": 50,      # at cap → F = 1.0
            "success_count": 100,
            "failure_count": 0,    # U ≈ 0.99
            "predictive": 1.0,
        }
        S = vault.compute_dars_score(payload, current_time=now)
        assert S > 0.9, f"Perfect memory should score > 0.9, got {S}"

    def test_stale_low_utility(self, vault):
        """Old memory with zero utility → S should be very low."""
        now = time.time()
        payload = {
            "recency": now - (30 * 86400),  # 30 days ago
            "frequency": 0,
            "success_count": 0,
            "failure_count": 5,
            "predictive": 0.0,
        }
        S = vault.compute_dars_score(payload, current_time=now)
        assert S < 0.3, f"Stale, useless memory should score < 0.3, got {S}"

    def test_score_clamped_to_unit(self, vault):
        """DARS score should always be in [0, 1]."""
        now = time.time()
        # Even with extreme values
        payload = {
            "recency": now,
            "frequency": 1000,
            "success_count": 10000,
            "failure_count": 0,
            "predictive": 10.0,  # intentionally out of range
        }
        S = vault.compute_dars_score(payload, current_time=now)
        assert 0.0 <= S <= 1.0

    def test_weights_influence(self, vault):
        """Changing only one parameter should shift the score predictably."""
        now = time.time()
        base = {
            "recency": now,
            "frequency": 10,
            "success_count": 5,
            "failure_count": 0,
            "predictive": 0.5,
        }
        base_score = vault.compute_dars_score(base, current_time=now)

        # Increase utility → score should increase
        high_util = {**base, "success_count": 50}
        high_score = vault.compute_dars_score(high_util, current_time=now)
        assert high_score > base_score

    def test_missing_payload_keys_use_defaults(self, vault):
        """Missing keys in payload should use safe defaults."""
        S = vault.compute_dars_score({}, current_time=time.time())
        assert isinstance(S, float)
        assert 0.0 <= S <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Retention Classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetentionClassification:
    """Tests for the DARS triage classifier."""

    def test_retain_threshold(self, vault):
        """Score > 0.7 → retain."""
        assert vault.classify_memory(0.8) == "retain"
        assert vault.classify_memory(0.71) == "retain"
        assert vault.classify_memory(1.0) == "retain"

    def test_compress_threshold(self, vault):
        """0.3 < Score ≤ 0.7 → compress."""
        assert vault.classify_memory(0.5) == "compress"
        assert vault.classify_memory(0.31) == "compress"
        assert vault.classify_memory(0.7) == "compress"

    def test_delete_threshold(self, vault):
        """Score ≤ 0.3 → delete."""
        assert vault.classify_memory(0.3) == "delete"
        assert vault.classify_memory(0.1) == "delete"
        assert vault.classify_memory(0.0) == "delete"

    def test_exact_boundaries(self, vault):
        """Verify exact boundary values."""
        assert vault.classify_memory(0.70) == "compress"  # ≤ 0.7
        assert vault.classify_memory(0.700001) == "retain"  # > 0.7
        assert vault.classify_memory(0.30) == "delete"      # ≤ 0.3
        assert vault.classify_memory(0.300001) == "compress" # > 0.3
