"""
LabOpt — Lab Optimisation Tool
================================

GUI entry point
---------------
    python main.py

Headless / scripting example
-----------------------------
This file shows how to use the backend modules without the GUI, useful for
batch processing or integration into scripts.  Replace the TODO sections with
your actual experiment runner.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Ensure UTF-8 output on Windows terminals (avoids cp1252 encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from csv_loader import load_csv, load_trials_from_csv
from optuna_builder import ask_batch, build_study, load_historical_trials, tell_batch
from parameter_config import (
    AllowedSubRange,
    ContextConfig,
    ObjectiveConfig,
    ParameterConfig,
    ParameterType,
    StudyConfig,
)
from session_manager import SessionManager


def run_headless_example() -> None:
    """
    Minimal headless example: seed 3 historical trials from a hard-coded
    dataset, ask for 2 more, then 'measure' them with a mock function.

    In a real lab workflow:
      1. Replace the mock data with a real CSV file.
      2. Replace the `mock_measure` function with your actual experiment runner.
      3. Use the GUI (python main.py) for the interactive version.
    """

    # ── 1. Define the search space ─────────────────────────────────────────
    config = StudyConfig(
        parameters=[
            ParameterConfig(
                name="spin_speed",
                ptype=ParameterType.INT,
                full_min=1000,
                full_max=5000,
                allowed_subranges=[
                    AllowedSubRange(1000, 2800),   # dead region: 2800–3200
                    AllowedSubRange(3200, 5000),
                ],
            ),
            ParameterConfig(
                name="concentration",
                ptype=ParameterType.FLOAT,
                full_min=0.01,
                full_max=0.15,
            ),
            ParameterConfig(
                name="hotplate_temp",
                ptype=ParameterType.INT,
                full_min=40,
                full_max=150,
            ),
            ParameterConfig(
                name="spin_time",
                ptype=ParameterType.INT,
                full_min=10,
                full_max=90,
            ),
        ],
        objectives=[ObjectiveConfig(column_name="thickness_nm", direction="maximize")],
        batch_size=2,
        n_batches=1,
        sampler_name="TPE",
    )

    # ── 2. Create a temporary session (uses a temp directory) ──────────────
    session_dir = tempfile.mkdtemp(prefix="labopt_headless_")
    state = SessionManager.create_new_session(config, csv_path="", session_dir=session_dir)
    study = build_study(config, state.storage_path, state.study_name)

    # ── 3. Seed 3 historical data points ──────────────────────────────────
    historical = [
        {"params": {"spin_speed": 2000, "concentration": 0.05, "hotplate_temp": 80,  "spin_time": 30}, "values": [125.3]},
        {"params": {"spin_speed": 3000, "concentration": 0.08, "hotplate_temp": 100, "spin_time": 45}, "values": [89.7]},
        {"params": {"spin_speed": 1500, "concentration": 0.03, "hotplate_temp": 60,  "spin_time": 60}, "values": [210.5]},
    ]
    added, skipped = load_historical_trials(study, historical, config)
    print(f"Seeded {added} historical trials ({skipped} skipped).")

    # ── 4. Ask for a batch ─────────────────────────────────────────────────
    trials = ask_batch(study, config, batch_size=config.batch_size)
    print(f"\nNext {len(trials)} suggested experiment(s):")
    for t in trials:
        print(f"  Trial {t.number}: {t.params}")

    # ── 5. Simulate measuring the result (replace with real lab measurement) ─
    def mock_measure(params: dict) -> float:
        """Toy model: higher spin speed → thinner film."""
        speed = params.get("spin_speed", 2000)
        conc  = params.get("concentration", 0.05)
        return 300.0 - speed * 0.04 + conc * 500

    results = [
        {"trial_number": t.number, "values": [mock_measure(t.params)]}
        for t in trials
    ]

    # ── 6. Tell the study ─────────────────────────────────────────────────
    tell_batch(study, [r["trial_number"] for r in results], [r["values"] for r in results])
    SessionManager.clear_pending_batch(state)

    from optuna.trial import TrialState
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\nStudy now has {len(completed)} completed trials.")
    print(f"Best trial: {study.best_trial.params}  →  {study.best_trial.value:.2f} nm")

    # Clean up temp files
    import shutil
    shutil.rmtree(session_dir, ignore_errors=True)


def run_contextual_example() -> None:
    """
    Example showing context variables — uncontrollable environmental conditions
    (e.g. ambient humidity) that the surrogate learns from without suggesting.

    The model trains on:
        f(spin_speed, concentration, humidity_pct) → thickness_nm

    At suggestion time, the user provides today's humidity so the optimizer
    can pick parameters that work best under those specific conditions.

    Context corrections
    -------------------
    After running experiments, the actual measured humidity is passed back via
    ``context_values`` in ``tell_batch()``.  This may differ from the planned
    humidity if conditions changed mid-experiment (e.g. a leak, rain, AC failure).
    """
    import math

    # ── 1. Define config with a context variable ────────────────────────────
    config = StudyConfig(
        parameters=[
            ParameterConfig(
                name="spin_speed",
                ptype=ParameterType.INT,
                full_min=1000,
                full_max=5000,
            ),
            ParameterConfig(
                name="concentration",
                ptype=ParameterType.FLOAT,
                full_min=0.01,
                full_max=0.15,
            ),
        ],
        objectives=[ObjectiveConfig(column_name="thickness_nm", direction="maximize")],
        context_variables=[
            ContextConfig(
                column_name="humidity_pct",
                description="Ambient humidity (%)",
            ),
        ],
        batch_size=2,
        n_batches=1,
        sampler_name="TPE",
    )

    session_dir = tempfile.mkdtemp(prefix="labopt_ctx_")
    state = SessionManager.create_new_session(config, csv_path="", session_dir=session_dir)
    study = build_study(config, state.storage_path, state.study_name)

    # ── 2. Seed historical data including context (humidity) values ──────────
    def mock_thickness(speed: int, conc: float, humidity: float) -> float:
        """Film quality degrades at high humidity."""
        return 300.0 - speed * 0.04 + conc * 500 - humidity * 0.5

    historical_with_ctx = [
        {
            "params": {"spin_speed": 2000, "concentration": 0.05},
            "values": [mock_thickness(2000, 0.05, 60.0)],
            "user_attrs": {"ctx_humidity_pct": 60.0},
        },
        {
            "params": {"spin_speed": 3000, "concentration": 0.08},
            "values": [mock_thickness(3000, 0.08, 72.0)],
            "user_attrs": {"ctx_humidity_pct": 72.0},
        },
        {
            "params": {"spin_speed": 1500, "concentration": 0.03},
            "values": [mock_thickness(1500, 0.03, 55.0)],
            "user_attrs": {"ctx_humidity_pct": 55.0},
        },
    ]
    added, _ = load_historical_trials(study, historical_with_ctx, config)
    print(f"Seeded {added} historical trials with humidity context.")

    # ── 3. Ask for suggestions conditioned on today's humidity ──────────────
    # In a real script, fetch this from a sensor, database, or API:
    #   current_humidity = fetch_from_sensor("humidity")
    current_humidity = 65.0   # example: today is moderately humid
    print(f"\nCurrent conditions: humidity = {current_humidity}%")

    trials = ask_batch(
        study, config, batch_size=config.batch_size,
        context={"humidity_pct": current_humidity},
    )
    print(f"Contextual suggestions ({len(trials)}):")
    for t in trials:
        print(f"  Trial {t.number}: {t.params}")

    # ── 4. Run experiments, measure results ──────────────────────────────────
    # During the experiment, actual humidity differed slightly from planned:
    actual_humidities = [67.2, 68.5]   # e.g. humidity crept up mid-experiment

    results = [
        {
            "trial_number": t.number,
            "values": [mock_thickness(
                t.params.get("spin_speed", 2000),
                t.params.get("concentration", 0.05),
                actual_humidities[i],
            )],
        }
        for i, t in enumerate(trials)
    ]

    # ── 5. Tell with ACTUAL context (corrected from planned) ─────────────────
    context_values = [
        {"humidity_pct": h} for h in actual_humidities
    ]
    tell_batch(
        study,
        [r["trial_number"] for r in results],
        [r["values"] for r in results],
        config=config,                         # required for context storage
        context_values=context_values,         # actual conditions, not planned
    )
    SessionManager.clear_pending_batch(state)

    from optuna.trial import TrialState
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\nTotal completed trials: {len(completed)}")

    # Verify context was stored correctly
    new_trials = [t for t in completed if t.number >= added]
    for t in new_trials:
        h = t.user_attrs.get("ctx_humidity_pct", "not stored")
        print(f"  Trial {t.number}: humidity_pct stored = {h}")

    import shutil
    shutil.rmtree(session_dir, ignore_errors=True)
    print("\nContext variable example complete.")


if __name__ == "__main__":
    print("Running headless LabOpt example...\n")
    run_headless_example()
    print("\n" + "─" * 60)
    print("Running contextual BO example (context variables)...\n")
    run_contextual_example()
    print("\nDone.  For the full GUI, run:  python main.py")
