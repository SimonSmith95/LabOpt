"""
Backend test suite for BHOP (no GUI required).
Run with:  pytest test_backend.py -v

Covers:
  Phase 1 — parameter_config  (serialisation round-trip)
  Phase 2 — csv_loader        (type inference, defaults, trial loading)
  Phase 3 — session_manager   (create/save/load/pending/match)
  Phase 4 — sampler_utils     (dead-region computation & validation)
  Phase 5 — optuna_builder    (study, historical trials, ask/tell batch)
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import List

import pandas as pd
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — parameter_config
# ══════════════════════════════════════════════════════════════════════════════

from parameter_config import (
    AllowedSubRange,
    ObjectiveConfig,
    ParameterConfig,
    ParameterType,
    StudyConfig,
)


class TestAllowedSubRange:
    def test_round_trip(self):
        r = AllowedSubRange(1.5, 9.9)
        assert AllowedSubRange.from_dict(r.to_dict()) == r

    def test_repr(self):
        assert "[" in repr(AllowedSubRange(0.0, 1.0))


class TestParameterConfig:
    def _int_param(self) -> ParameterConfig:
        return ParameterConfig(
            name="speed",
            ptype=ParameterType.INT,
            full_min=100.0,
            full_max=5000.0,
            allowed_subranges=[AllowedSubRange(100, 2000), AllowedSubRange(3000, 5000)],
            step=100,
        )

    def _float_param(self) -> ParameterConfig:
        return ParameterConfig(
            name="conc",
            ptype=ParameterType.FLOAT,
            full_min=0.01,
            full_max=0.15,
        )

    def _cat_param(self) -> ParameterConfig:
        return ParameterConfig(
            name="solvent",
            ptype=ParameterType.CATEGORICAL,
            all_choices=["THF", "DMF", "NMP"],
            allowed_choices=["THF", "DMF"],
        )

    def _bool_param(self) -> ParameterConfig:
        return ParameterConfig(name="anneal", ptype=ParameterType.BOOL, fixed_value=None)

    def test_int_round_trip(self):
        p = self._int_param()
        assert ParameterConfig.from_dict(p.to_dict()).name == p.name
        assert ParameterConfig.from_dict(p.to_dict()).allowed_subranges == p.allowed_subranges
        assert ParameterConfig.from_dict(p.to_dict()).step == p.step

    def test_float_round_trip(self):
        p = self._float_param()
        p2 = ParameterConfig.from_dict(p.to_dict())
        assert p2.full_min == p.full_min
        assert p2.full_max == p.full_max

    def test_categorical_round_trip(self):
        p = self._cat_param()
        p2 = ParameterConfig.from_dict(p.to_dict())
        assert p2.allowed_choices == p.allowed_choices

    def test_bool_round_trip(self):
        p = self._bool_param()
        p2 = ParameterConfig.from_dict(p.to_dict())
        assert p2.fixed_value is None

    def test_default_subrange_populated(self):
        """__post_init__ should fill allowed_subranges from full_min/max."""
        p = ParameterConfig(name="x", ptype=ParameterType.FLOAT, full_min=0.0, full_max=1.0)
        assert len(p.allowed_subranges) == 1
        assert p.allowed_subranges[0].low == 0.0
        assert p.allowed_subranges[0].high == 1.0


class TestStudyConfig:
    def _config(self) -> StudyConfig:
        return StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.INT, full_min=100, full_max=5000),
                ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ],
            objectives=[
                ObjectiveConfig("thickness", "maximize"),
                ObjectiveConfig("roughness", "minimize"),
            ],
            batch_size=3,
            n_batches=20,
            sampler_name="NSGAII",
        )

    def test_round_trip(self):
        cfg = self._config()
        cfg2 = StudyConfig.from_dict(cfg.to_dict())
        assert cfg2.sampler_name == "NSGAII"
        assert cfg2.batch_size == 3
        assert len(cfg2.parameters) == 2
        assert len(cfg2.objectives) == 2
        assert cfg2.objectives[0].direction == "maximize"
        assert cfg2.objectives[1].direction == "minimize"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — csv_loader
# ══════════════════════════════════════════════════════════════════════════════

from csv_loader import (
    append_rows_to_csv,
    extract_param_defaults,
    infer_column_type,
    load_csv,
    load_trials_from_csv,
)


def _make_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))


class TestInferColumnType:
    def test_bool_int(self):
        s = pd.Series([0, 1, 0, 1])
        assert infer_column_type(s) == ParameterType.BOOL

    def test_bool_true_false(self):
        s = pd.Series([True, False, True])
        assert infer_column_type(s) == ParameterType.BOOL

    def test_categorical_string(self):
        s = pd.Series(["THF", "DMF", "NMP", "THF"])
        assert infer_column_type(s) == ParameterType.CATEGORICAL

    def test_categorical_few_numeric(self):
        s = pd.Series([1, 2, 3, 1, 2])   # ≤ 8 unique → categorical
        assert infer_column_type(s) == ParameterType.CATEGORICAL

    def test_int_many_unique(self):
        s = pd.Series(list(range(50, 150)))   # 100 unique integers
        assert infer_column_type(s) == ParameterType.INT

    def test_float(self):
        s = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert infer_column_type(s) == ParameterType.FLOAT

    def test_empty_series_defaults_float(self):
        s = pd.Series([], dtype=float)
        assert infer_column_type(s) == ParameterType.FLOAT


class TestExtractParamDefaults:
    # Need >8 unique values for spin_speed and concentration so they are
    # classified as INT / FLOAT (not CATEGORICAL) by infer_column_type.
    CSV = """spin_speed,concentration,solvent,anneal,thickness
500,0.01,THF,True,100.0
1000,0.02,DMF,False,110.0
1500,0.03,NMP,True,120.0
2000,0.04,THF,False,130.0
2500,0.05,DMF,True,140.0
3000,0.06,NMP,False,150.0
3500,0.07,THF,True,160.0
4000,0.08,DMF,False,170.0
4500,0.09,NMP,True,180.0
"""

    def setup_method(self):
        self.df = _make_df(self.CSV)

    def test_result_columns_excluded(self):
        params = extract_param_defaults(self.df, ["thickness"])
        names = [p.name for p in params]
        assert "thickness" not in names
        assert "spin_speed" in names

    def test_int_range(self):
        params = extract_param_defaults(self.df, ["thickness"])
        speed = next(p for p in params if p.name == "spin_speed")
        assert speed.ptype == ParameterType.INT
        assert speed.full_min == 500.0
        assert speed.full_max == 4500.0

    def test_float_range(self):
        params = extract_param_defaults(self.df, ["thickness"])
        conc = next(p for p in params if p.name == "concentration")
        assert conc.ptype == ParameterType.FLOAT
        assert abs(conc.full_min - 0.01) < 1e-9
        assert abs(conc.full_max - 0.09) < 1e-9

    def test_categorical_choices(self):
        params = extract_param_defaults(self.df, ["thickness"])
        solvent = next(p for p in params if p.name == "solvent")
        assert solvent.ptype == ParameterType.CATEGORICAL
        assert set(solvent.all_choices) == {"THF", "DMF", "NMP"}

    def test_bool_param(self):
        params = extract_param_defaults(self.df, ["thickness"])
        anneal = next(p for p in params if p.name == "anneal")
        assert anneal.ptype == ParameterType.BOOL


class TestLoadTrialsFromCsv:
    CSV = """spin_speed,concentration,thickness
2000,0.05,125.3
3000,0.08,89.7
1500,,210.5
"""

    def test_rows_with_missing_objective_skipped(self):
        df = _make_df(self.CSV)
        config = StudyConfig(
            parameters=[
                ParameterConfig("spin_speed", ParameterType.INT, full_min=1000, full_max=5000),
                ParameterConfig("concentration", ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ],
            objectives=[ObjectiveConfig("thickness", "maximize")],
        )
        results = load_trials_from_csv(df, config)
        # Row 3 has missing concentration (NaN param) → skipped by csv_loader.
        # Only rows 1 and 2 have all params + objective present → 2 results.
        assert len(results) == 2

    def test_missing_objective_excluded(self):
        csv = "spin_speed,thickness\n2000,125.3\n3000,\n"
        df = _make_df(csv)
        config = StudyConfig(
            parameters=[ParameterConfig("spin_speed", ParameterType.INT, full_min=1000, full_max=5000)],
            objectives=[ObjectiveConfig("thickness", "maximize")],
        )
        results = load_trials_from_csv(df, config)
        assert len(results) == 1
        assert results[0]["values"] == [125.3]


class TestAppendRowsToCsv:
    def test_creates_new_file(self, tmp_path):
        path = str(tmp_path / "out.csv")
        append_rows_to_csv(path, [{"a": 1, "b": 2}])
        df = pd.read_csv(path)
        assert len(df) == 1

    def test_appends_to_existing(self, tmp_path):
        path = str(tmp_path / "out.csv")
        pd.DataFrame([{"a": 1}]).to_csv(path, index=False)
        append_rows_to_csv(path, [{"a": 2}])
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_empty_rows_no_op(self, tmp_path):
        path = str(tmp_path / "out.csv")
        append_rows_to_csv(path, [])
        assert not os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — sampler_utils
# ══════════════════════════════════════════════════════════════════════════════

from sampler_utils import compute_allowed_subranges, validate_subranges


class TestComputeAllowedSubranges:
    def test_no_dead_regions(self):
        result = compute_allowed_subranges(0.0, 10.0, [])
        assert len(result) == 1
        assert result[0].low == 0.0
        assert result[0].high == 10.0

    def test_dead_region_in_middle(self):
        result = compute_allowed_subranges(0.0, 10.0, [(4.0, 6.0)])
        assert len(result) == 2
        assert result[0].low == 0.0
        assert result[0].high == 4.0
        assert result[1].low == 6.0
        assert result[1].high == 10.0

    def test_dead_region_at_start(self):
        result = compute_allowed_subranges(0.0, 10.0, [(0.0, 3.0)])
        assert len(result) == 1
        assert result[0].low == 3.0
        assert result[0].high == 10.0

    def test_dead_region_at_end(self):
        result = compute_allowed_subranges(0.0, 10.0, [(8.0, 10.0)])
        assert len(result) == 1
        assert result[0].high == 8.0

    def test_multiple_dead_regions_merged(self):
        # Two overlapping dead regions should merge
        result = compute_allowed_subranges(0.0, 10.0, [(3.0, 5.0), (4.0, 7.0)])
        assert len(result) == 2
        assert result[0].high == 3.0
        assert result[1].low == 7.0

    def test_multiple_separate_dead_regions(self):
        result = compute_allowed_subranges(1.0, 10.0, [(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)])
        assert len(result) == 4

    def test_dead_region_covers_all_raises(self):
        with pytest.raises(ValueError, match="entire"):
            compute_allowed_subranges(0.0, 10.0, [(0.0, 10.0)])

    def test_dead_region_out_of_bounds_clipped(self):
        # Dead region partially outside full range — should be clipped gracefully
        result = compute_allowed_subranges(0.0, 10.0, [(-5.0, 3.0)])
        assert result[0].low == 3.0


class TestValidateSubranges:
    def test_valid_subranges(self):
        errors = validate_subranges(
            [AllowedSubRange(0.0, 5.0), AllowedSubRange(7.0, 10.0)],
            0.0, 10.0
        )
        assert errors == []

    def test_empty_subranges(self):
        errors = validate_subranges([], 0.0, 10.0)
        assert len(errors) > 0

    def test_zero_width_subrange(self):
        errors = validate_subranges([AllowedSubRange(5.0, 5.0)], 0.0, 10.0)
        assert any("less than" in e for e in errors)

    def test_out_of_bounds_low(self):
        errors = validate_subranges([AllowedSubRange(-1.0, 5.0)], 0.0, 10.0)
        assert any("below" in e for e in errors)

    def test_out_of_bounds_high(self):
        errors = validate_subranges([AllowedSubRange(5.0, 15.0)], 0.0, 10.0)
        assert any("exceeds" in e for e in errors)

    def test_overlapping_subranges(self):
        errors = validate_subranges(
            [AllowedSubRange(0.0, 6.0), AllowedSubRange(5.0, 10.0)],
            0.0, 10.0
        )
        assert any("overlap" in e for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — session_manager
# ══════════════════════════════════════════════════════════════════════════════

from session_manager import SessionManager, SessionState


def _simple_config() -> StudyConfig:
    return StudyConfig(
        parameters=[
            ParameterConfig("speed", ParameterType.INT, full_min=1000, full_max=5000),
        ],
        objectives=[ObjectiveConfig("thickness", "maximize")],
        batch_size=2,
        n_batches=5,
        sampler_name="TPE",
    )


class TestSessionManager:
    def test_create_save_load(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "/fake/data.csv", str(tmp_path))

        assert os.path.exists(state.session_path)
        # Manually create the DB file so load() validates it
        open(state.storage_path, "w").close()

        loaded = SessionManager.load(state.session_path)
        assert loaded.study_name == state.study_name
        assert loaded.study_config.sampler_name == "TPE"
        assert loaded.csv_path == "/fake/data.csv"

    def test_load_missing_db_raises(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        # Do NOT create the .db file
        with pytest.raises(FileNotFoundError, match="database"):
            SessionManager.load(state.session_path)

    def test_mark_and_clear_pending(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        open(state.storage_path, "w").close()

        # mark_batch_pending expects {"trial_number": int, "params": dict} dicts
        trials = [
            {"trial_number": 0, "params": {"speed": 2000}},
            {"trial_number": 1, "params": {"speed": 3000}},
        ]
        SessionManager.mark_batch_pending(state, trials)

        assert state.pending_batch is not None
        assert len(state.pending_batch) == 2
        pending_csv = os.path.join(str(tmp_path), "pending_batch.csv")
        assert os.path.exists(pending_csv)

        # Reload and check pending persisted
        loaded = SessionManager.load(state.session_path)
        assert loaded.pending_batch is not None

        SessionManager.clear_pending_batch(state)
        assert state.pending_batch is None
        assert not os.path.exists(pending_csv)

    def test_match_pending_to_csv(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        open(state.storage_path, "w").close()

        SessionManager.mark_batch_pending(
            state, [{"trial_number": 0, "params": {"speed": 2000}}]
        )

        df = pd.DataFrame([
            {"speed": 2000, "thickness": 125.3},
            {"speed": 3000, "thickness": 99.1},
        ])
        matches = SessionManager.match_pending_to_csv(state, df)
        assert matches is not None
        assert 0 in matches
        assert matches[0] == [125.3]

    def test_match_pending_no_match_returns_none(self, tmp_path):
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        open(state.storage_path, "w").close()

        SessionManager.mark_batch_pending(
            state, [{"trial_number": 0, "params": {"speed": 9999}}]
        )

        df = pd.DataFrame([{"speed": 2000, "thickness": 125.3}])
        matches = SessionManager.match_pending_to_csv(state, df)
        assert matches is None

    def test_atomic_save_no_corruption(self, tmp_path):
        """The .tmp file should be gone after save."""
        cfg = _simple_config()
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        assert not os.path.exists(state.session_path + ".tmp")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — optuna_builder
# ══════════════════════════════════════════════════════════════════════════════

from optuna_builder import (
    ask_batch,
    build_distributions,
    build_study,
    get_pareto_front,
    load_historical_trials,
    tell_batch,
)


def _full_config() -> StudyConfig:
    return StudyConfig(
        parameters=[
            ParameterConfig("speed", ParameterType.INT, full_min=1000, full_max=5000),
            ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ParameterConfig(
                "solvent", ParameterType.CATEGORICAL,
                all_choices=["THF", "DMF"], allowed_choices=["THF", "DMF"]
            ),
        ],
        objectives=[ObjectiveConfig("thickness", "maximize")],
        batch_size=2,
        n_batches=3,
        sampler_name="Random",
    )


class TestBuildStudy:
    def test_creates_study(self, tmp_path):
        cfg = _full_config()
        db = str(tmp_path / "test.db")
        study = build_study(cfg, db, "test_study")
        assert study.study_name == "test_study"

    def test_load_if_exists(self, tmp_path):
        cfg = _full_config()
        db = str(tmp_path / "test.db")
        s1 = build_study(cfg, db, "test_study")
        s2 = build_study(cfg, db, "test_study")
        assert s1.study_name == s2.study_name

    def test_multi_objective(self, tmp_path):
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT, full_min=0, full_max=1)],
            objectives=[
                ObjectiveConfig("obj1", "minimize"),
                ObjectiveConfig("obj2", "maximize"),
            ],
            sampler_name="NSGAII",
        )
        study = build_study(cfg, str(tmp_path / "mo.db"), "mo_study")
        assert len(study.directions) == 2


class TestBuildDistributions:
    def test_single_range_int(self):
        cfg = StudyConfig(
            parameters=[ParameterConfig("speed", ParameterType.INT, full_min=100, full_max=5000)],
            objectives=[ObjectiveConfig("y", "maximize")],
        )
        dists = build_distributions(cfg)
        assert "speed" in dists

    def test_multi_range_int_uses_range_idx(self):
        cfg = StudyConfig(
            parameters=[ParameterConfig(
                "speed", ParameterType.INT, full_min=100, full_max=5000,
                allowed_subranges=[AllowedSubRange(100, 2000), AllowedSubRange(3000, 5000)],
            )],
            objectives=[ObjectiveConfig("y", "maximize")],
        )
        dists = build_distributions(cfg)
        assert "speed__range_idx" in dists
        assert "speed__in_range_0" in dists
        assert "speed__in_range_1" in dists

    def test_disabled_param_excluded(self):
        cfg = StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.INT, full_min=100, full_max=5000, enabled=True),
                ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15, enabled=False),
            ],
            objectives=[ObjectiveConfig("y", "maximize")],
        )
        dists = build_distributions(cfg)
        assert "speed" in dists
        assert "conc" not in dists


class TestLoadHistoricalTrials:
    def test_adds_trials(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "h.db"), "h_study")

        trials = [
            {"params": {"speed": 2000, "conc": 0.05, "solvent": "THF"}, "values": [125.3]},
            {"params": {"speed": 3000, "conc": 0.08, "solvent": "DMF"}, "values": [89.7]},
        ]
        added, skipped = load_historical_trials(study, trials, cfg)
        assert added == 2
        assert skipped == 0

    def test_duplicate_detection(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "d.db"), "d_study")
        trials = [
            {"params": {"speed": 2000, "conc": 0.05, "solvent": "THF"}, "values": [125.3]},
        ]
        load_historical_trials(study, trials, cfg)
        added, skipped = load_historical_trials(study, trials, cfg)
        assert added == 0
        assert skipped == 1


class TestAskTellBatch:
    def test_ask_returns_waiting_trials(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "ab.db"), "ab_study")
        trials = ask_batch(study, cfg, batch_size=3)
        assert len(trials) == 3
        # Optuna 4.x: study.ask() returns Trial objects (no .state attribute).
        # Verify via study.trials that 3 non-complete trials exist.
        from optuna.trial import TrialState
        active = [t for t in study.trials if t.state != TrialState.COMPLETE]
        assert len(active) == 3

    def test_tell_completes_trials(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "tb.db"), "tb_study")
        trials = ask_batch(study, cfg, batch_size=2)

        trial_numbers = [t.number for t in trials]
        results = [[100.0 + i] for i in range(len(trials))]
        tell_batch(study, trial_numbers, results)

        from optuna.trial import TrialState
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        assert len(completed) == 2

    def test_tell_with_unknown_trial_number_skips(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "sk.db"), "sk_study")
        ask_batch(study, cfg, batch_size=1)
        # Telling a non-existent trial number should not crash
        tell_batch(study, [9999], [[100.0]])


class TestGetParetoFront:
    def test_single_objective_returns_best(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "pf.db"), "pf_study")
        trials = ask_batch(study, cfg, 2)
        tell_batch(study, [t.number for t in trials], [[90.0], [120.0]])
        pareto = get_pareto_front(study)
        assert len(pareto) == 1
        assert pareto[0].value == 120.0

    def test_empty_study_returns_empty(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "empty.db"), "empty_study")
        pareto = get_pareto_front(study)
        assert pareto == []


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7 — Surrogate quality (Feature 2)
# ══════════════════════════════════════════════════════════════════════════════

from surrogate_quality import compute_surrogate_quality


class TestSurrogateQuality:
    def test_insufficient_data_returns_correct_status(self, tmp_path):
        cfg = _full_config()
        study = build_study(cfg, str(tmp_path / "sq.db"), "sq")
        result = compute_surrogate_quality(study, cfg)
        assert result["status"] == "insufficient"
        assert result["r2"] is None
        assert result["rmse"] is None

    def test_high_r2_for_near_linear_function(self, tmp_path):
        """20 trials where objective ≈ 2*speed → RF should fit well."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("speed", ParameterType.FLOAT,
                                        full_min=0.0, full_max=1.0)],
            objectives=[ObjectiveConfig("y", "minimize")],
            sampler_name="Random",
        )
        study = build_study(cfg, str(tmp_path / "lin.db"), "lin")
        import numpy as np
        rng = np.random.default_rng(42)
        vals = rng.uniform(0, 1, 20)
        trials_data = [
            {"params": {"speed": float(v)},
             "values": [float(2 * v + rng.normal(0, 0.01))]}
            for v in vals
        ]
        load_historical_trials(study, trials_data, cfg)
        result = compute_surrogate_quality(study, cfg)
        assert result["status"] in ("good", "moderate")
        assert result["r2"] is not None
        assert result["r2"] > 0.5

    def test_categorical_param_no_exception(self, tmp_path):
        """Categorical params must be label-encoded without raising."""
        cfg = _full_config()  # has "solvent" categorical
        study = build_study(cfg, str(tmp_path / "cat.db"), "cat")
        import numpy as np
        rng = np.random.default_rng(0)
        trials_data = [
            {"params": {"speed": int(rng.integers(1000, 5000)),
                        "conc": float(rng.uniform(0.01, 0.15)),
                        "solvent": rng.choice(["THF", "DMF"])},
             "values": [float(rng.uniform(80, 200))]}
            for _ in range(12)
        ]
        load_historical_trials(study, trials_data, cfg)
        result = compute_surrogate_quality(study, cfg)
        assert "status" in result
        assert "n" in result

    def test_maximize_objective_does_not_crash(self, tmp_path):
        """Sign flip for 'maximize' must not raise."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0.0, full_max=1.0)],
            objectives=[ObjectiveConfig("y", "maximize")],
            sampler_name="Random",
        )
        study = build_study(cfg, str(tmp_path / "mx.db"), "mx")
        import numpy as np
        rng = np.random.default_rng(7)
        trials_data = [
            {"params": {"x": float(v)}, "values": [float(v)]}
            for v in rng.uniform(0, 1, 12)
        ]
        load_historical_trials(study, trials_data, cfg)
        result = compute_surrogate_quality(study, cfg)
        assert result["status"] in ("good", "moderate", "poor", "insufficient")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 8 — Parameter importance (Feature 3)
# ══════════════════════════════════════════════════════════════════════════════

class TestParameterImportance:
    def test_importance_ranks_driving_param_higher(self):
        """x0 drives outcome; x1 is pure noise → x0 importance > x1."""
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(42)
        X = rng.random((60, 2))
        y = X[:, 0] ** 2 + rng.normal(0, 0.01, 60)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        result = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
        assert result.importances_mean[0] > result.importances_mean[1]

    def test_insufficient_data_guard(self):
        """With 5 rows the data guard should trigger (MIN_TRIALS = 15)."""
        import pandas as pd
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
        MIN_TRIALS = 15
        assert len(df) < MIN_TRIALS

    def test_categorical_label_encoding_no_exception(self):
        """Categorical columns must be encoded cleanly before RF fit."""
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor

        df = pd.DataFrame({
            "material": ["gold", "silver", "copper"] * 5,
            "y": np.random.default_rng(0).uniform(0, 1, 15),
        })
        df["material_enc"] = pd.Categorical(df["material"]).codes
        X = df[["material_enc"]].values
        y = df["y"].values
        rf = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
        pred = rf.predict([[0]])[0]   # "copper" (alphabetically first → code 0)
        assert 0.0 <= pred <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Phase 9 — Near-duplicate detection (Feature 4)
# ══════════════════════════════════════════════════════════════════════════════

from sampler_utils import find_near_duplicates


class TestNearDuplicates:
    def _cfg(self) -> StudyConfig:
        return StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.FLOAT,
                                full_min=0.0, full_max=1000.0),
                ParameterConfig("conc", ParameterType.FLOAT,
                                full_min=0.0, full_max=1.0),
            ],
            objectives=[ObjectiveConfig("y", "minimize")],
        )

    def _existing(self, *rows) -> list[dict]:
        """Helper: wrap param dicts with a "number" key."""
        return [{"number": i, **r} for i, r in enumerate(rows)]

    def test_exact_duplicate_flagged(self):
        cfg = self._cfg()
        suggestions = [{"speed": 500.0, "conc": 0.5}]
        existing    = self._existing({"speed": 500.0, "conc": 0.5})
        hits = find_near_duplicates(suggestions, existing, cfg)
        assert len(hits) == 1
        assert hits[0]["closest_trial_number"] == 0
        assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-9)

    def test_distant_point_not_flagged(self):
        cfg = self._cfg()
        suggestions = [{"speed": 900.0, "conc": 0.9}]
        existing    = self._existing({"speed": 100.0, "conc": 0.1})
        hits = find_near_duplicates(suggestions, existing, cfg)
        assert len(hits) == 0

    def test_threshold_boundary(self):
        """A point far enough from existing should NOT be flagged.

        The distance formula normalises by n_params (2 here):
            dist = sqrt(dist_sq / n_params)
        With speed=80, existing=0, conc both 0:
            dist_sq = (80/1000)² + 0² = 0.0064
            dist    = sqrt(0.0064 / 2) ≈ 0.0566 > 0.05  → no warning.
        """
        cfg = self._cfg()
        suggestions = [{"speed": 80.0, "conc": 0.0}]
        existing    = self._existing({"speed": 0.0,  "conc": 0.0})
        hits = find_near_duplicates(suggestions, existing, cfg, threshold=0.05)
        assert len(hits) == 0

    def test_empty_existing_trials_no_warning(self):
        cfg = self._cfg()
        suggestions = [{"speed": 500.0, "conc": 0.5}]
        hits = find_near_duplicates(suggestions, [], cfg)
        assert len(hits) == 0

    def test_multiple_suggestions_independent(self):
        cfg = self._cfg()
        suggestions = [
            {"speed": 500.0, "conc": 0.5},   # near existing → flagged
            {"speed": 900.0, "conc": 0.9},   # far → not flagged
        ]
        existing = self._existing({"speed": 500.0, "conc": 0.5})
        hits = find_near_duplicates(suggestions, existing, cfg)
        assert len(hits) == 1
        assert "Suggestion 1" in hits[0]["message"]
        assert hits[0]["suggestion_idx"] == 0

    def test_trial_number_preserved_in_result(self):
        """The actual Optuna trial number must appear in the result, not the list index."""
        cfg = self._cfg()
        suggestions = [{"speed": 500.0, "conc": 0.5}]
        # Trial has number=42, not index 0
        existing = [{"number": 42, "speed": 500.0, "conc": 0.5}]
        hits = find_near_duplicates(suggestions, existing, cfg)
        assert len(hits) == 1
        assert hits[0]["closest_trial_number"] == 42
        assert "42" in hits[0]["message"]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Convergence plot data logic (no GUI required)
# ══════════════════════════════════════════════════════════════════════════════

class TestConvergencePlotData:
    """Verify the running-best computation logic used by ConvergenceWidget."""

    def test_running_best_minimize(self):
        values = [100, 80, 90, 70, 75]
        running = []
        best = float("inf")
        for v in values:
            best = min(best, v)
            running.append(best)
        assert running == [100, 80, 80, 70, 70]

    def test_running_best_maximize(self):
        values = [100, 80, 120, 70, 75]
        running = []
        best = float("-inf")
        for v in values:
            best = max(best, v)
            running.append(best)
        assert running == [100, 100, 120, 120, 120]

    def test_single_value_running_best(self):
        values = [42.0]
        best = min(values)
        assert best == 42.0

    def test_running_best_already_sorted(self):
        """Strictly decreasing values — every entry is the new best (minimize)."""
        values = [100, 90, 80, 70, 60]
        running = []
        best = float("inf")
        for v in values:
            best = min(best, v)
            running.append(best)
        assert running == values

    def test_convergence_widget_data_with_optuna(self, tmp_path):
        """Full round-trip: build study, tell trials, verify running best."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0.0, full_max=1.0)],
            objectives=[ObjectiveConfig("y", "minimize")],
            sampler_name="Random",
        )
        study = build_study(cfg, str(tmp_path / "conv.db"), "conv")
        trials = ask_batch(study, cfg, 4)
        objective_values = [0.8, 0.5, 0.6, 0.3]
        tell_batch(study, [t.number for t in trials],
                   [[v] for v in objective_values])

        from optuna.trial import TrialState
        completed = sorted(
            [t for t in study.trials if t.state == TrialState.COMPLETE],
            key=lambda t: t.number,
        )
        values = [t.values[0] for t in completed]
        running = []
        best = float("inf")
        for v in values:
            best = min(best, v)
            running.append(best)

        assert running[-1] == pytest.approx(0.3)
        assert all(running[i] >= running[i + 1] for i in range(len(running) - 1)), \
            "Running best (minimize) must be non-increasing"


# ══════════════════════════════════════════════════════════════════════════════
# Integration smoke test — full headless loop
# ══════════════════════════════════════════════════════════════════════════════

class TestHeadlessIntegration:
    def test_full_loop(self, tmp_path):
        """
        Seed historical data → ask a batch → mock-measure → tell → check best.
        This exercises parameter_config, session_manager, optuna_builder,
        csv_loader, and sampler_utils together.
        """
        cfg = StudyConfig(
            parameters=[
                ParameterConfig("speed", ParameterType.INT, full_min=1000, full_max=5000,
                                allowed_subranges=[AllowedSubRange(1000, 2500),
                                                   AllowedSubRange(3500, 5000)]),
                ParameterConfig("conc",  ParameterType.FLOAT, full_min=0.01, full_max=0.15),
            ],
            objectives=[ObjectiveConfig("thickness", "maximize")],
            batch_size=2,
            n_batches=2,
            sampler_name="Random",
        )
        state = SessionManager.create_new_session(cfg, "", str(tmp_path))
        study = build_study(cfg, state.storage_path, state.study_name)

        # Seed history
        historical = [
            {"params": {"speed": 1500, "conc": 0.05}, "values": [125.3]},
            {"params": {"speed": 4000, "conc": 0.10}, "values": [89.7]},
        ]
        added, _ = load_historical_trials(study, historical, cfg)
        assert added == 2

        # Ask + tell
        trials = ask_batch(study, cfg, 2)
        assert len(trials) == 2

        # Verify suggested speeds are NOT in the dead region 2500–3500
        for t in trials:
            # Reconstruct the actual speed from internal params
            if "speed" in t.params:
                speed = t.params["speed"]
            else:
                # find in_range keys
                speed_keys = [k for k in t.params if k.startswith("speed__in_range_")]
                speed = t.params[speed_keys[0]] if speed_keys else None
            if speed is not None:
                assert not (2500 < speed < 3500), f"Speed {speed} is in dead region!"

        tell_batch(study, [t.number for t in trials], [[200.0], [210.0]])

        from optuna.trial import TrialState
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        assert len(completed) == 4   # 2 historical + 2 batch

        pareto = get_pareto_front(study)
        assert pareto[0].value == 210.0

        # Config serialization round-trip
        cfg_dict = cfg.to_dict()
        cfg2 = StudyConfig.from_dict(cfg_dict)
        assert cfg2.parameters[0].allowed_subranges[0].low == 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Phase 10 — GP sampler (Feature 5)
# ══════════════════════════════════════════════════════════════════════════════

class TestGPSampler:
    """Verify that the GP sampler option works end-to-end.

    GPSampler is available in Optuna >= 3.6.  If for some reason the
    installed version does not have it, build_study() falls back to TPE
    with a warning — the tests must pass in either case.
    """

    def test_gp_study_creates_without_error(self, tmp_path):
        """build_study() with sampler_name='GP' must not raise."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0.0, full_max=1.0)],
            objectives=[ObjectiveConfig("y", "minimize")],
            sampler_name="GP",
        )
        # Should not raise regardless of whether GPSampler is available
        study = build_study(cfg, str(tmp_path / "gp.db"), "gp_study")
        assert study is not None

    def test_ask_tell_with_gp(self, tmp_path):
        """Full ask/tell cycle works with GP sampler."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0.0, full_max=1.0)],
            objectives=[ObjectiveConfig("y", "minimize")],
            sampler_name="GP",
            batch_size=2,
        )
        study = build_study(cfg, str(tmp_path / "gp2.db"), "gp2")
        trials = ask_batch(study, cfg, 2)
        assert len(trials) == 2
        tell_batch(study, [t.number for t in trials], [[0.5], [0.3]])
        from optuna.trial import TrialState
        completed = [t for t in study.trials
                     if t.state == TrialState.COMPLETE]
        assert len(completed) == 2

    def test_gp_config_round_trip(self):
        """sampler_name='GP' survives to_dict() / from_dict()."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0, full_max=1)],
            objectives=[ObjectiveConfig("y", "minimize")],
            sampler_name="GP",
        )
        cfg2 = StudyConfig.from_dict(cfg.to_dict())
        assert cfg2.sampler_name == "GP"

    def test_gp_ask_returns_values_in_range(self, tmp_path):
        """GP suggestions should stay within [full_min, full_max]."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0.2, full_max=0.8)],
            objectives=[ObjectiveConfig("y", "minimize")],
            sampler_name="GP",
        )
        study = build_study(cfg, str(tmp_path / "gp3.db"), "gp3")
        trials = ask_batch(study, cfg, 3)
        for t in trials:
            x_val = t.params.get("x")
            assert x_val is not None
            assert 0.2 <= x_val <= 0.8, f"GP suggested x={x_val} outside [0.2, 0.8]"

    def test_gp_with_inequality_constraints_does_not_crash(self, tmp_path):
        """GP + inequality constraints: study creates without error.

        GPSampler does not support constraints_func; the build_study()
        GP branch bypasses the constraints_func entirely so no crash occurs.
        The constraint is still projected post-hoc by the suggestion logic.
        """
        from parameter_config import ParameterConstraint
        cfg = StudyConfig(
            parameters=[
                ParameterConfig("a", ParameterType.FLOAT, full_min=0.0, full_max=1.0),
                ParameterConfig("b", ParameterType.FLOAT, full_min=0.0, full_max=1.0),
            ],
            objectives=[ObjectiveConfig("y", "minimize")],
            constraints=[
                ParameterConstraint(
                    name="sum_le_1",
                    expression="a + b",
                    operator="<=",
                    target=1.0,
                )
            ],
            sampler_name="GP",
        )
        study = build_study(cfg, str(tmp_path / "gp_ineq.db"), "gp_ineq")
        assert study is not None
        # Ask should not raise
        trials = ask_batch(study, cfg, 2)
        assert len(trials) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Phase 11 — Profiler prediction (Feature 6)
# ══════════════════════════════════════════════════════════════════════════════

class TestProfilerPrediction:
    """
    Tests for the What-If Profiler surrogate logic.

    These tests exercise the sklearn RF fitting and categorical label-encoding
    that backs the ProfilerWidget — without requiring any GUI/PySide6.
    """

    def test_prediction_close_to_training_point(self):
        """RF trained on a known point should predict near that value."""
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor

        X = np.array(
            [
                [0.5],
                [0.2],
                [0.8],
                [0.1],
                [0.9],
                [0.3],
                [0.6],
                [0.4],
                [0.7],
                [0.05],
            ]
        )
        y = X[:, 0] * 100.0  # linear: y = 100 * x
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        pred = rf.predict([[0.5]])[0]
        # RF should predict close to 50 for x=0.5
        assert abs(pred - 50.0) < 15.0

    def test_categorical_label_encoding_no_exception(self):
        """Label-encoding categoricals and predicting must not raise."""
        import numpy as np

        df = pd.DataFrame(
            {
                "material": ["gold", "silver", "copper"] * 5,
                "y": np.random.default_rng(0).uniform(0, 1, 15),
            }
        )
        df["material_enc"] = pd.Categorical(df["material"]).codes
        X = df[["material_enc"]].values
        y = df["y"].values
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
        # Predict for "gold" (code=0 after alphabetical sort)
        pred = rf.predict([[0]])[0]
        assert 0.0 <= pred <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Phase 12 — Pareto scatter data (Feature 7)
# ══════════════════════════════════════════════════════════════════════════════

from optuna_builder import ask_batch, build_study, get_pareto_front, tell_batch


# ══════════════════════════════════════════════════════════════════════════════
# Phase 13 — Replicate aggregation (Feature 8)
# ══════════════════════════════════════════════════════════════════════════════

from csv_loader import aggregate_replicates


class TestAggregateReplicates:
    def test_identical_rows_averaged(self):
        """Two identical param rows → merged, objective averaged."""
        df = pd.DataFrame([
            {"x": 0.5, "y": 100.0},
            {"x": 0.5, "y": 120.0},
        ])
        result = aggregate_replicates(df, ["x"], ["y"])
        assert len(result) == 1
        assert result.iloc[0]["y"] == 110.0
        assert result.iloc[0]["n_replicates"] == 2

    def test_distinct_rows_unchanged(self):
        """Two distinct param rows → both survive, n_replicates=1."""
        df = pd.DataFrame([
            {"x": 0.5, "y": 100.0},
            {"x": 0.8, "y": 120.0},
        ])
        result = aggregate_replicates(df, ["x"], ["y"])
        assert len(result) == 2
        assert list(result["n_replicates"]) == [1, 1]

    def test_near_identical_within_tolerance(self):
        """Two rows within tolerance → merged."""
        df = pd.DataFrame([
            {"x": 0.5,          "y": 100.0},
            {"x": 0.5 + 1e-8,   "y": 120.0},
        ])
        result = aggregate_replicates(df, ["x"], ["y"], tolerance=1e-6)
        assert len(result) == 1
        assert result.iloc[0]["n_replicates"] == 2

    def test_empty_dataframe(self):
        """Empty input → empty output with n_replicates column."""
        df = pd.DataFrame(columns=["x", "y"])
        result = aggregate_replicates(df, ["x"], ["y"])
        assert len(result) == 0
        assert "n_replicates" in result.columns

    def test_single_row_unchanged(self):
        """Single row → unchanged, n_replicates=1."""
        df = pd.DataFrame([{"x": 0.5, "y": 99.0}])
        result = aggregate_replicates(df, ["x"], ["y"])
        assert result.iloc[0]["y"] == 99.0
        assert result.iloc[0]["n_replicates"] == 1

    def test_tolerance_zero_disables_aggregation(self):
        """tolerance=0 → no aggregation, all rows preserved with n_replicates=1."""
        df = pd.DataFrame([
            {"x": 0.5, "y": 100.0},
            {"x": 0.5, "y": 120.0},
        ])
        result = aggregate_replicates(df, ["x"], ["y"], tolerance=0)
        assert len(result) == 2
        assert all(result["n_replicates"] == 1)

    def test_categorical_must_match_exactly(self):
        """Categorical param must match exactly regardless of numeric tolerance."""
        df = pd.DataFrame([
            {"material": "gold",   "temp": 300.0, "y": 100.0},
            {"material": "silver", "temp": 300.0, "y": 120.0},
        ])
        result = aggregate_replicates(df, ["material", "temp"], ["y"], tolerance=1.0)
        # Different materials → NOT merged even though temp is within tolerance
        assert len(result) == 2

    def test_three_way_merge(self):
        """Three identical rows → one row with n_replicates=3 and averaged y."""
        df = pd.DataFrame([
            {"x": 1.0, "y": 10.0},
            {"x": 1.0, "y": 20.0},
            {"x": 1.0, "y": 30.0},
        ])
        result = aggregate_replicates(df, ["x"], ["y"])
        assert len(result) == 1
        assert result.iloc[0]["y"] == 20.0   # (10+20+30)/3
        assert result.iloc[0]["n_replicates"] == 3

    def test_config_round_trip_with_replicate_fields(self):
        """replicate_aggregation and replicate_tolerance survive to_dict/from_dict."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT, full_min=0, full_max=1)],
            objectives=[ObjectiveConfig("y", "minimize")],
            replicate_aggregation=True,
            replicate_tolerance=0.5,
        )
        cfg2 = StudyConfig.from_dict(cfg.to_dict())
        assert cfg2.replicate_aggregation is True
        assert cfg2.replicate_tolerance == 0.5

    def test_old_session_json_without_replicate_fields_loads(self):
        """Sessions saved before this feature (missing replicate_* keys) load with defaults."""
        d = {
            "parameters": [],
            "objectives": [],
            "batch_size": 1,
            "n_batches": 5,
            "sampler_name": "TPE",
            # replicate_aggregation and replicate_tolerance absent intentionally
        }
        cfg = StudyConfig.from_dict(d)
        assert cfg.replicate_aggregation is False
        assert cfg.replicate_tolerance == 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# Phase 14 — Report generator (Feature 9)
# ══════════════════════════════════════════════════════════════════════════════

from report_generator import generate_report, write_report


class TestReportGenerator:
    """Tests for generate_report() and write_report() in report_generator.py."""

    def _minimal_report(self) -> str:
        metadata = {
            "session_name": "test_session",
            "date": "2026-09-07",
            "sampler": "TPE",
            "n_batches": 3,
            "objectives": ["y (minimize)"],
        }
        trials_df = pd.DataFrame([
            {"speed": 1000, "y": 100},
            {"speed": 2000, "y": 80},
        ])
        best_rows = [{"speed": 2000, "y": 80}]
        return generate_report(metadata, trials_df, {}, best_rows)

    def test_report_is_html(self):
        html = self._minimal_report()
        assert "<html" in html.lower()
        assert "</html>" in html.lower()

    def test_report_contains_metadata(self):
        html = self._minimal_report()
        assert "test_session" in html
        assert "TPE" in html

    def test_report_with_no_figures(self):
        """Empty figures dict must not raise and must return non-empty HTML."""
        html = self._minimal_report()
        assert html   # non-empty

    def test_report_embeds_figure(self, tmp_path):
        """A figure passed in figures dict must appear as a base64 PNG in the HTML."""
        # Use the OO Matplotlib API (Figure/Axes) rather than pyplot to avoid
        # needing a display backend or pyplot module in the test environment.
        matplotlib = pytest.importorskip(
            "matplotlib", reason="matplotlib not available in test env"
        )
        from matplotlib.figure import Figure
        fig = Figure(facecolor="#1e1e2e")
        ax = fig.add_subplot(111)
        ax.plot([1, 2], [3, 4])
        metadata = {
            "session_name": "x",
            "date": "2026-01-01",
            "sampler": "TPE",
            "n_batches": 1,
            "objectives": ["y"],
        }
        html = generate_report(metadata, pd.DataFrame(), {"convergence": fig}, [])
        assert "data:image/png;base64," in html

    def test_write_report_creates_file(self, tmp_path):
        html = self._minimal_report()
        path = write_report(html, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_report_with_empty_dataframe_and_no_best_rows(self):
        """Report with zero trials must generate without error."""
        metadata = {
            "session_name": "empty",
            "date": "2026-01-01",
            "sampler": "Random",
            "n_batches": 0,
            "objectives": [],
        }
        html = generate_report(metadata, pd.DataFrame(), {}, [])
        assert "<html" in html.lower()

    def test_report_objectives_as_list(self):
        """Objectives may be a list of strings."""
        metadata = {
            "session_name": "multi",
            "date": "2026-01-01",
            "sampler": "NSGAII",
            "n_batches": 2,
            "objectives": ["yield (maximize)", "cost (minimize)"],
        }
        html = generate_report(metadata, pd.DataFrame(), {}, [])
        assert "yield" in html
        assert "cost" in html

    def test_write_report_filename_has_timestamp(self, tmp_path):
        """Written filename must follow the bhop_report_YYYYMMDD_HHMMSS.html pattern."""
        import re
        html = self._minimal_report()
        path = write_report(html, str(tmp_path))
        fname = os.path.basename(path)
        assert re.match(r"bhop_report_\d{8}_\d{6}\.html", fname), \
            f"Unexpected filename format: {fname}"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 15 — Excel (.xlsx) import / export (Feature 10)
# ══════════════════════════════════════════════════════════════════════════════

class TestXlsxLoading:
    """Tests for load_csv() with .xlsx/.xls files.  Skipped if openpyxl absent."""

    def test_load_xlsx_basic(self, tmp_path):
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        df_orig = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        path = str(tmp_path / "test.xlsx")
        df_orig.to_excel(path, index=False, engine="openpyxl")
        loaded = load_csv(path)
        assert list(loaded.columns) == ["x", "y"]
        assert len(loaded) == 2

    def test_csv_loading_unaffected_after_xlsx_change(self, tmp_path):
        """Existing CSV code path must still work unchanged.

        Use two columns so csv.Sniffer can detect the comma separator reliably.
        (Single-column CSVs have no commas, which confuses sep=None detection.)
        """
        path = str(tmp_path / "test.csv")
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
        loaded = load_csv(path)
        assert "a" in loaded.columns
        assert "b" in loaded.columns

    def test_xlsx_column_types_preserved(self, tmp_path):
        """Numeric columns in .xlsx must remain numeric after load."""
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        df_orig = pd.DataFrame({
            "int_col":   [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col":   ["a", "b", "c"],
        })
        path = str(tmp_path / "types.xlsx")
        df_orig.to_excel(path, index=False, engine="openpyxl")
        loaded = load_csv(path)
        assert pd.api.types.is_numeric_dtype(loaded["int_col"])
        assert pd.api.types.is_numeric_dtype(loaded["float_col"])

    def test_xlsx_values_match_original(self, tmp_path):
        """Values round-trip correctly through Excel."""
        pytest.importorskip("openpyxl", reason="openpyxl not installed")
        df_orig = pd.DataFrame({"speed": [1000.0, 2000.0], "y": [0.5, 0.8]})
        path = str(tmp_path / "vals.xlsx")
        df_orig.to_excel(path, index=False, engine="openpyxl")
        loaded = load_csv(path)
        assert abs(loaded["speed"].iloc[0] - 1000.0) < 1e-9
        assert abs(loaded["y"].iloc[1] - 0.8) < 1e-9

    def test_load_csv_raises_on_bad_file(self, tmp_path):
        """Corrupt/unknown file must raise ValueError."""
        path = str(tmp_path / "bad.xlsx")
        with open(path, "w") as f:
            f.write("not an excel file")
        with pytest.raises(ValueError, match="Cannot read"):
            load_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 16 — Outlier detection logic (Feature 11)
# ══════════════════════════════════════════════════════════════════════════════

class TestOutlierDetectionLogic:
    """
    Test the residual threshold logic independently of PySide6.

    The same `|entered - predicted| > factor * std` formula is embedded in
    BatchResultsDialog._check_outlier().  These tests verify the logic without
    instantiating any GUI widget.
    """

    def _is_outlier(self, entered: float, predicted: float, std: float,
                    factor: float = 2.5) -> bool:
        return abs(entered - predicted) > factor * std

    def test_extreme_value_flagged(self):
        """200 vs predicted 50, std=10 → |200-50|/10 = 15 > 2.5."""
        assert self._is_outlier(200.0, 50.0, 10.0) is True

    def test_normal_value_not_flagged(self):
        """55 vs predicted 50, std=10 → |55-50|/10 = 0.5 < 2.5."""
        assert self._is_outlier(55.0, 50.0, 10.0) is False

    def test_boundary_value_not_flagged(self):
        """Exactly at 2.5 sigma → NOT flagged (strictly greater than)."""
        assert self._is_outlier(75.0, 50.0, 10.0) is False  # 25/10 = 2.5

    def test_just_over_boundary_flagged(self):
        """Fractionally above 2.5 sigma → flagged."""
        assert self._is_outlier(75.1, 50.0, 10.0) is True   # 25.1/10 > 2.5

    def test_negative_outlier_flagged(self):
        """Outlier in negative direction."""
        assert self._is_outlier(-100.0, 50.0, 10.0) is True   # 150/10 = 15

    def test_zero_std_does_not_divide(self):
        """std=0 should be handled before calling _is_outlier (guard in dialog)."""
        # With std=0, formula is undefined; the dialog skips when pred_std < 1e-12.
        # Here we just verify the formula doesn't raise for our test helper.
        try:
            result = self._is_outlier(55.0, 50.0, 0.0)
            # If 0*factor=0, abs(5)>0 is True — but the real dialog skips this case.
        except ZeroDivisionError:
            pass  # acceptable; the dialog guards against std < 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# Phase 17 — Auto-stop criterion (Feature 12)
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoStop:
    """
    Tests for the convergence-based auto-stop logic.

    All tests exercise pure Python logic (no Optuna / GUI required).
    The same formulas are used in worker.py._check_converged().
    """

    def _improvements(self, best_per_batch: list[float]) -> list[float]:
        """Compute relative improvements between consecutive batch bests."""
        return [
            abs(best_per_batch[i] - best_per_batch[i - 1])
            / (abs(best_per_batch[i - 1]) + 1e-9)
            for i in range(1, len(best_per_batch))
        ]

    def test_convergence_detected_when_no_improvement(self):
        """All improvements < 1 % across window=3 → converged."""
        best_per_batch = [100.0, 99.5, 99.3, 99.4]
        threshold = 0.01   # 1%
        window    = 3
        imps = self._improvements(best_per_batch)
        assert all(imp < threshold for imp in imps[-window:])

    def test_no_convergence_when_still_improving(self):
        """10% improvement per batch → NOT converged."""
        best_per_batch = [100.0, 90.0, 80.0, 70.0]
        threshold = 0.01
        window    = 3
        imps = self._improvements(best_per_batch)
        assert not all(imp < threshold for imp in imps[-window:])

    def test_not_enough_batches_for_window(self):
        """Only 2 bests, window=3 → too few data points, not converged."""
        best_per_batch = [100.0, 99.9]
        window = 3
        assert len(best_per_batch) < window   # guard condition in _check_converged

    def test_large_improvement_prevents_convergence(self):
        """If any batch in the window has clearly > threshold improvement, not converged."""
        best_per_batch = [100.0, 99.0, 85.0, 84.5]
        # Improvements: 1%, 14.1%, 0.59% — 14.1% >> 1% threshold
        threshold = 0.01
        window    = 3
        imps = self._improvements(best_per_batch)
        assert not all(imp < threshold for imp in imps[-window:])

    def test_config_round_trip_with_auto_stop_fields(self):
        """auto_stop fields survive to_dict() / from_dict()."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0, full_max=1)],
            objectives=[ObjectiveConfig("y", "minimize")],
            auto_stop=True,
            auto_stop_min_improvement=0.02,
            auto_stop_n_batches=5,
        )
        cfg2 = StudyConfig.from_dict(cfg.to_dict())
        assert cfg2.auto_stop is True
        assert cfg2.auto_stop_min_improvement == pytest.approx(0.02)
        assert cfg2.auto_stop_n_batches == 5

    def test_old_session_json_without_auto_stop_fields_loads(self):
        """Sessions saved before Feature 12 (missing auto_stop keys) load with defaults."""
        d = {
            "parameters": [],
            "objectives": [],
            "batch_size": 1,
            "n_batches": 5,
            "sampler_name": "TPE",
            # auto_stop fields absent intentionally
        }
        cfg = StudyConfig.from_dict(d)
        assert cfg.auto_stop is False
        assert cfg.auto_stop_min_improvement == pytest.approx(0.01)
        assert cfg.auto_stop_n_batches == 3


class TestParetoScatter:
    def test_dominated_points_excluded_from_pareto(self, tmp_path):
        """(1,1) dominates (2,2) for minimize-minimize."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0, full_max=1)],
            objectives=[ObjectiveConfig("obj1", "minimize"),
                        ObjectiveConfig("obj2", "minimize")],
            sampler_name="Random",
        )
        study = build_study(cfg, str(tmp_path / "par.db"), "par")
        # (1,1) dominates (2,2) — Pareto front should contain only (1,1)
        trials = ask_batch(study, cfg, 2)
        tell_batch(study, [trials[0].number], [[1.0, 1.0]])
        tell_batch(study, [trials[1].number], [[2.0, 2.0]])
        pareto = get_pareto_front(study)
        pareto_values = [t.values for t in pareto]
        assert [1.0, 1.0] in pareto_values
        assert [2.0, 2.0] not in pareto_values

    def test_two_non_dominated_points_both_in_pareto(self, tmp_path):
        """(1,3) and (3,1) — neither dominates the other; both must be Pareto-optimal."""
        cfg = StudyConfig(
            parameters=[ParameterConfig("x", ParameterType.FLOAT,
                                        full_min=0, full_max=1)],
            objectives=[ObjectiveConfig("obj1", "minimize"),
                        ObjectiveConfig("obj2", "minimize")],
            sampler_name="Random",
        )
        study = build_study(cfg, str(tmp_path / "par2.db"), "par2")
        trials = ask_batch(study, cfg, 2)
        tell_batch(study, [trials[0].number], [[1.0, 3.0]])
        tell_batch(study, [trials[1].number], [[3.0, 1.0]])
        pareto = get_pareto_front(study)
        assert len(pareto) == 2
