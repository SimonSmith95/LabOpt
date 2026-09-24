"""
Tests for power_analysis_widget.py
====================================
All pure-math tests (compute_required_n, achieved_power) run without Qt.
Widget smoke-tests use a minimal QApplication fixture.

Run with:
    python -m pytest test_power_analysis.py -v
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest

# Pure-math functions: no Qt dependency — import from the Qt-free module.
from power_analysis_math import achieved_power, compute_required_n


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the entire test session (no display required)."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(["test", "--platform", "offscreen"])
    return app


# ══════════════════════════════════════════════════════════════════════════════
# compute_required_n — one-sample / paired tests
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRequiredN:
    """Tests for the pure-function compute_required_n."""

    # ── Classic textbook values ──────────────────────────────────────────────

    def test_medium_effect_one_sample(self):
        """d=0.5, α=0.05, power=0.80 → N ≈ 32 (classic result)."""
        result = compute_required_n(sigma=10.0, delta=5.0, alpha=0.05, power=0.80)
        assert result["n"] == 32, f"Expected 32, got {result['n']}"

    def test_large_effect_one_sample(self):
        """d=0.8, α=0.05, power=0.80 → N ≈ 13."""
        result = compute_required_n(sigma=10.0, delta=8.0, alpha=0.05, power=0.80)
        # Classic result is 13 or 14 depending on rounding convention
        assert 12 <= result["n"] <= 15, f"Expected ~13, got {result['n']}"

    def test_small_effect_one_sample(self):
        """d=0.2, α=0.05, power=0.80 → N ≈ 197."""
        result = compute_required_n(sigma=10.0, delta=2.0, alpha=0.05, power=0.80)
        assert 193 <= result["n"] <= 201, f"Expected ~197, got {result['n']}"

    def test_very_large_effect_one_sample(self):
        """d=2.0, α=0.05, power=0.80 → small N."""
        result = compute_required_n(sigma=5.0, delta=10.0, alpha=0.05, power=0.80)
        assert result["n"] <= 10, f"Expected small N (≤10), got {result['n']}"

    # ── Two-sample formula ────────────────────────────────────────────────────

    def test_two_sample_roughly_doubles_n(self):
        """
        Two-sample N per group should be approximately double the one-sample N
        (exact factor is 2 in the formula, before ceiling).
        """
        one = compute_required_n(sigma=10.0, delta=5.0, alpha=0.05, power=0.80,
                                 two_sample=False)
        two = compute_required_n(sigma=10.0, delta=5.0, alpha=0.05, power=0.80,
                                 two_sample=True)
        # two_sample N (per group) should be approximately 2 × one_sample N
        assert two["n"] >= one["n"] * 1.8, (
            f"Two-sample N ({two['n']}) should be ≥ 1.8× one-sample N ({one['n']})"
        )

    def test_two_sample_symmetric(self):
        """Two-sample result should be strictly larger than one-sample."""
        one = compute_required_n(sigma=4.0, delta=2.0, two_sample=False)
        two = compute_required_n(sigma=4.0, delta=2.0, two_sample=True)
        assert two["n"] > one["n"]

    # ── Alpha / power sensitivity ─────────────────────────────────────────────

    def test_stricter_alpha_needs_more_n(self):
        """α=0.01 should require more experiments than α=0.05."""
        loose = compute_required_n(sigma=10.0, delta=5.0, alpha=0.05, power=0.80)
        strict = compute_required_n(sigma=10.0, delta=5.0, alpha=0.01, power=0.80)
        assert strict["n"] > loose["n"], (
            f"α=0.01 ({strict['n']}) should need more than α=0.05 ({loose['n']})"
        )

    def test_higher_power_needs_more_n(self):
        """power=0.95 should require more experiments than power=0.80."""
        low_power = compute_required_n(sigma=10.0, delta=5.0, alpha=0.05, power=0.80)
        high_power = compute_required_n(sigma=10.0, delta=5.0, alpha=0.05, power=0.95)
        assert high_power["n"] > low_power["n"], (
            f"power=0.95 ({high_power['n']}) should need more than "
            f"power=0.80 ({low_power['n']})"
        )

    # ── Minimum N guard ───────────────────────────────────────────────────────

    def test_minimum_n_is_two(self):
        """Even a huge effect size must return N ≥ 2."""
        result = compute_required_n(sigma=1.0, delta=1000.0)
        assert result["n"] >= 2, f"N must be at least 2, got {result['n']}"

    # ── Cohen's d values ──────────────────────────────────────────────────────

    def test_cohens_d_is_ratio(self):
        """Cohen's d must equal delta / sigma exactly."""
        result = compute_required_n(sigma=4.0, delta=2.0)
        assert math.isclose(result["cohens_d"], 0.5, rel_tol=1e-9)

    def test_cohens_d_classification_small(self):
        result = compute_required_n(sigma=10.0, delta=1.0)  # d=0.1
        assert result["effect_tag"] == "Small"

    def test_cohens_d_classification_medium(self):
        result = compute_required_n(sigma=10.0, delta=3.0)  # d=0.3
        assert result["effect_tag"] == "Medium"

    def test_cohens_d_classification_large(self):
        result = compute_required_n(sigma=10.0, delta=6.0)  # d=0.6
        assert result["effect_tag"] == "Large"

    def test_cohens_d_classification_very_large(self):
        result = compute_required_n(sigma=10.0, delta=9.0)  # d=0.9
        assert result["effect_tag"] == "Very large"

    def test_cohens_d_boundary_medium(self):
        """d=0.2 is the boundary between Small and Medium → Medium."""
        result = compute_required_n(sigma=10.0, delta=2.0)  # d=0.2 exactly
        assert result["effect_tag"] == "Medium"

    def test_cohens_d_boundary_large(self):
        """d=0.5 is the boundary between Medium and Large → Large."""
        result = compute_required_n(sigma=10.0, delta=5.0)  # d=0.5 exactly
        assert result["effect_tag"] == "Large"

    def test_cohens_d_boundary_very_large(self):
        """d=0.8 is the boundary between Large and Very large → Very large."""
        result = compute_required_n(sigma=10.0, delta=8.0)  # d=0.8 exactly
        assert result["effect_tag"] == "Very large"

    # ── Return-type contract ──────────────────────────────────────────────────

    def test_return_keys(self):
        result = compute_required_n(sigma=5.0, delta=2.5)
        assert set(result.keys()) == {"n", "cohens_d", "effect_tag"}

    def test_n_is_int(self):
        result = compute_required_n(sigma=5.0, delta=2.5)
        assert isinstance(result["n"], int)

    def test_cohens_d_is_float(self):
        result = compute_required_n(sigma=5.0, delta=2.5)
        assert isinstance(result["cohens_d"], float)

    # ── Monotonicity ──────────────────────────────────────────────────────────

    def test_larger_delta_needs_fewer_experiments(self):
        """Bigger effect → easier to detect → fewer experiments needed."""
        small_delta = compute_required_n(sigma=10.0, delta=2.0)
        large_delta = compute_required_n(sigma=10.0, delta=5.0)
        assert small_delta["n"] > large_delta["n"]

    def test_larger_sigma_needs_more_experiments(self):
        """More noise → harder to detect → more experiments needed."""
        low_noise = compute_required_n(sigma=5.0, delta=2.0)
        high_noise = compute_required_n(sigma=10.0, delta=2.0)
        assert high_noise["n"] > low_noise["n"]

    # ── Input validation ──────────────────────────────────────────────────────

    def test_raises_on_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            compute_required_n(sigma=0.0, delta=1.0)

    def test_raises_on_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            compute_required_n(sigma=-1.0, delta=1.0)

    def test_raises_on_zero_delta(self):
        with pytest.raises(ValueError, match="delta"):
            compute_required_n(sigma=1.0, delta=0.0)

    def test_raises_on_negative_delta(self):
        with pytest.raises(ValueError, match="delta"):
            compute_required_n(sigma=1.0, delta=-1.0)

    def test_raises_on_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            compute_required_n(sigma=1.0, delta=0.5, alpha=0.0)

    def test_raises_on_alpha_one(self):
        with pytest.raises(ValueError, match="alpha"):
            compute_required_n(sigma=1.0, delta=0.5, alpha=1.0)

    def test_raises_on_power_zero(self):
        with pytest.raises(ValueError, match="power"):
            compute_required_n(sigma=1.0, delta=0.5, power=0.0)

    def test_raises_on_power_one(self):
        with pytest.raises(ValueError, match="power"):
            compute_required_n(sigma=1.0, delta=0.5, power=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# achieved_power
# ══════════════════════════════════════════════════════════════════════════════

class TestAchievedPower:
    """Tests for the achieved_power inverse function."""

    def test_returns_float_in_unit_interval(self):
        pwr = achieved_power(n=32, sigma=10.0, delta=5.0)
        assert 0.0 <= pwr <= 1.0

    def test_power_at_required_n_approx_target(self):
        """At the required N the achieved power should be ≈ the target power."""
        target = 0.80
        result = compute_required_n(sigma=10.0, delta=5.0, power=target)
        pwr = achieved_power(result["n"], sigma=10.0, delta=5.0)
        # Ceiling means we might be slightly above the target
        assert pwr >= target - 0.01, (
            f"Achieved power {pwr:.4f} should be ≥ target {target}"
        )

    def test_power_increases_with_n(self):
        """More experiments → higher power."""
        pwr_small = achieved_power(n=10,  sigma=10.0, delta=5.0)
        pwr_large = achieved_power(n=100, sigma=10.0, delta=5.0)
        assert pwr_large > pwr_small

    def test_power_increases_with_delta(self):
        """Larger effect → easier to detect → higher power for same N."""
        pwr_small_d = achieved_power(n=30, sigma=10.0, delta=2.0)
        pwr_large_d = achieved_power(n=30, sigma=10.0, delta=6.0)
        assert pwr_large_d > pwr_small_d

    def test_two_sample_lower_power_than_one_sample(self):
        """For the same N, two-sample has lower power than one-sample."""
        one = achieved_power(n=32, sigma=10.0, delta=5.0, two_sample=False)
        two = achieved_power(n=32, sigma=10.0, delta=5.0, two_sample=True)
        assert one > two

    def test_very_large_n_gives_power_near_one(self):
        pwr = achieved_power(n=10_000, sigma=10.0, delta=1.0)
        assert pwr > 0.99


# ══════════════════════════════════════════════════════════════════════════════
# Widget tests (require QApplication — offscreen platform)
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Skip the entire widget test class when PySide6 is unavailable / broken
# (e.g. in a headless CI environment without a working Qt installation).
# ---------------------------------------------------------------------------
_pyside6_available = True
try:
    from PySide6.QtWidgets import QApplication as _QA  # noqa: F401
except Exception:
    _pyside6_available = False

_widget_skip = pytest.mark.skipif(
    not _pyside6_available,
    reason="PySide6 not available or Qt DLL broken in this environment",
)


@_widget_skip
class TestPowerAnalysisWidget:
    """Smoke tests for the PowerAnalysisWidget Qt class."""

    def test_widget_instantiates(self, qapp):
        """Widget must construct without raising."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        assert w is not None

    def test_initial_n_label(self, qapp):
        """After construction the N label should show a number, not '—'."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        text = w._n_label.text()
        assert text.startswith("N ="), f"Unexpected label: {text!r}"
        # Should contain a positive integer
        n_str = text.split("=")[-1].strip()
        assert n_str.isdigit() and int(n_str) >= 2

    def test_delta_spin_triggers_recalculate(self, qapp):
        """Changing Δ should update the N label."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._delta_spin.setValue(5.0)
        w._sigma_spin.setValue(10.0)
        n_before = int(w._n_label.text().split("=")[-1].strip())

        w._delta_spin.setValue(2.0)   # smaller effect → more N needed
        n_after = int(w._n_label.text().split("=")[-1].strip())
        assert n_after > n_before, (
            f"Smaller Δ should increase N: {n_before} → {n_after}"
        )

    def test_sigma_spin_triggers_recalculate(self, qapp):
        """Changing σ should update the N label."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._delta_spin.setValue(5.0)
        w._sigma_spin.setValue(5.0)
        n_low_sigma = int(w._n_label.text().split("=")[-1].strip())

        w._sigma_spin.setValue(15.0)   # more noise → more N needed
        n_high_sigma = int(w._n_label.text().split("=")[-1].strip())
        assert n_high_sigma > n_low_sigma

    def test_set_objective_data_enables_from_data_btn(self, qapp):
        """set_objective_data with ≥2 values should enable the 'From data' button."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        assert not w._from_data_btn.isEnabled()

        series = pd.Series([10.0, 12.0, 9.5, 11.2, 10.8])
        w.set_objective_data(series, n_current=0)
        assert w._from_data_btn.isEnabled()

    def test_set_objective_data_none_disables_from_data_btn(self, qapp):
        """set_objective_data(None) should disable the 'From data' button."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        series = pd.Series([1.0, 2.0, 3.0])
        w.set_objective_data(series)
        assert w._from_data_btn.isEnabled()

        w.set_objective_data(None)
        assert not w._from_data_btn.isEnabled()

    def test_sigma_estimation_from_series(self, qapp):
        """The estimated sigma should equal pd.Series.std(ddof=1)."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        data = pd.Series([10.0, 12.0, 9.5, 11.2, 10.8, 13.1, 9.0])
        expected_sigma = float(data.std(ddof=1))

        w.set_objective_data(data, n_current=0)
        assert w._sigma_from_data is not None
        assert math.isclose(w._sigma_from_data, expected_sigma, rel_tol=1e-9)

    def test_from_data_button_sets_sigma_spin(self, qapp):
        """Clicking 'From data' should copy the estimated σ into the spinbox."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        data = pd.Series([5.0, 7.0, 6.0, 8.0, 5.5])
        w.set_objective_data(data, n_current=0)
        expected_sigma = float(data.std(ddof=1))

        w._from_data_btn.click()
        assert math.isclose(w._sigma_spin.value(), expected_sigma, rel_tol=1e-6)

    def test_set_objective_data_single_value_no_crash(self, qapp):
        """A single value (std undefined) should not raise — button stays disabled."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w.set_objective_data(pd.Series([42.0]), n_current=0)
        assert not w._from_data_btn.isEnabled()

    def test_set_objective_data_all_nan_no_crash(self, qapp):
        """All-NaN series should not raise."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w.set_objective_data(pd.Series([float("nan"), float("nan")]), n_current=0)
        assert not w._from_data_btn.isEnabled()

    def test_badge_no_data(self, qapp):
        """n_current=0 → badge text mentions no trial data."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w.set_objective_data(None, n_current=0)
        assert "No trial data" in w._badge_label.text() or "no trial" in w._badge_label.text().lower()

    def test_badge_adequate(self, qapp):
        """n_current ≥ required_n → badge text contains ✅."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._delta_spin.setValue(5.0)
        w._sigma_spin.setValue(10.0)
        required_n = compute_required_n(10.0, 5.0)["n"]
        w.set_objective_data(None, n_current=required_n + 5)
        assert "✅" in w._badge_label.text()

    def test_badge_underpowered(self, qapp):
        """n_current < 50% of required_n → badge text contains 🔴."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._delta_spin.setValue(5.0)
        w._sigma_spin.setValue(10.0)
        required_n = compute_required_n(10.0, 5.0)["n"]
        w.set_objective_data(None, n_current=max(1, required_n // 4))
        assert "🔴" in w._badge_label.text()

    def test_badge_borderline(self, qapp):
        """n_current ≈ 70% of required_n → badge text contains ⚠."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._delta_spin.setValue(5.0)
        w._sigma_spin.setValue(10.0)
        required_n = compute_required_n(10.0, 5.0)["n"]
        borderline_n = int(required_n * 0.7)
        # Ensure borderline_n is in the 50-99% range
        if borderline_n < max(1, required_n // 2):
            borderline_n = max(1, required_n // 2) + 1
        w.set_objective_data(None, n_current=borderline_n)
        assert "⚠" in w._badge_label.text()

    def test_two_sample_label_shows_per_group(self, qapp):
        """Switching to two-sample should show 'per group' in the sub-label."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._test_combo.setCurrentIndex(1)   # "Two-sample (independent)"
        assert "per group" in w._n_sublabel.text()

    def test_one_sample_label_no_per_group(self, qapp):
        """One-sample label should not mention 'per group'."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._test_combo.setCurrentIndex(0)   # "One-sample / paired"
        assert "per group" not in w._n_sublabel.text()

    def test_cohens_label_shows_effect_tag(self, qapp):
        """Cohen's label must include the effect tag string."""
        from power_analysis_widget import PowerAnalysisWidget
        w = PowerAnalysisWidget()
        w._delta_spin.setValue(5.0)
        w._sigma_spin.setValue(10.0)   # d=0.5 → "Large"
        assert "Large" in w._cohens_label.text()
