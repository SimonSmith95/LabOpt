# LabOpt — Lab Optimisation Tool

> Bayesian optimisation for wet-lab and materials-science teams.  
> Load a CSV, define your objectives, and let the model suggest the next batch of experiments — no ML expertise required.

---

## What is it?

LabOpt wraps the [Optuna](https://optuna.org/) optimisation framework in a clean PySide6 GUI designed for real laboratory workflows:

- Experiments are run in **batches** — you ask the model for *N* suggestions, run them in the lab, enter the results, and repeat.
- Your data lives in a **CSV file** you already have (historical data, an experiment log, or a blank template).
- Sessions **persist** across days and weeks — close the app, come back, and carry on exactly where you left off.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

**Requires Python 3.10+.** See [§ Installation & Setup](DOCUMENTATION.md#2-installation--setup) for the full dependency list.

### 2. Launch the GUI

```bash
python main.py
```

### 3. The basic loop

1. **File → Load CSV** — point LabOpt at your experiment data.
2. **Configure parameters** — set ranges, mark dead regions, fix constants.
3. **Set objectives** — pick which columns to minimise or maximise.
4. **Ask Next Batch** — the model suggests the next experiments.
5. **Run your experiments** in the lab.
6. **Enter results** in the dialog — or import from a CSV.
7. Repeat from step 4.

> 📖 Full GUI walkthrough → [§ 3 How to Use the Application](DOCUMENTATION.md#3-how-to-use-the-application-gui-walkthrough)

---

## Key Features

| Feature | Details |
|---|---|
| **Bayesian optimisation** | TPE, GP, NSGA-II, and Random samplers via Optuna |
| **Multi-objective** | Pareto-front optimisation with interactive scatter plot |
| **Parameter constraints** | Algebraic equality & inequality rules (e.g. fractions summing to 1) |
| **Dead regions** | Forbidden parameter sub-ranges the model never suggests |
| **Context variables** | Uncontrollable environmental conditions (humidity, pressure) that the surrogate learns from without being suggested — corrections allowed after the experiment |
| **Session persistence** | SQLite + JSON — resume any session from any machine |
| **Design space visualisation** | Pairplot / parallel coordinates / 1-D marginals with batch overlay |
| **Correlation matrix** | Pearson or Spearman heatmap across all parameters and objectives |
| **Feature importance** | Permutation importance from a Random Forest surrogate |
| **What-if Profiler** | Real-time predicted objective as you slide parameter values — context variable sliders included |
| **Surrogate quality badge** | Cross-validated R² with traffic-light rating (✅ ⚠ 🔴) |
| **Outlier detection** | Flags implausible result entries before you submit them |
| **Replicate aggregation** | Merge near-duplicate rows and average their objective values |
| **Auto-stop** | Halt automatically when the best value has converged |
| **Headless / scripting API** | Run the full loop from Python with no GUI — context values accepted from any source (database, sensor, CSV) |
| **Power Analysis** | Sample-size calculator: given measurement spread σ and minimum detectable effect Δ, compute required N — with live power-curve and N-vs-Δ plots |
| **Surrogate Validation** | Rigorous §0–§5 checks: data quality, RF + GP hold-out and cross-validated accuracy, virtual BO benchmark (GP+EI vs RF+EI vs Greedy vs Random), Expected Improvement marginals, pass/fail summary. Triggered from the Results tab checkbox. Results embedded in the HTML report and shown as tabs in the Design Space dialog. Requires ≥ 30 unique rows. |

---

## Sampler Guide

| Sampler | Best for | Notes |
|---|---|---|
| `TPE` | Single-objective, ≤ 20 parameters | Default. Multivariate Parzen estimator. |
| `GP` | Smooth spaces, < 200 trials | Gaussian Process. Most sample-efficient; scales as O(n³). |
| `NSGAII` | Multi-objective | Genetic algorithm. Needs larger batch sizes to converge. |
| `Random` | Baselines / debugging | No model — uniform random sampling. |

> 📖 Sampler details → [§ 3.5 Choosing Batch Size, n-Batches and Sampler](DOCUMENTATION.md#35-choosing-batch-size-n-batches-and-sampler)

---

## Parameter Constraints

LabOpt supports algebraic constraints that suggested experiments must satisfy:

```
CsPbI + FAPbI + MAPbI = 1.0    ← equality (residual param auto-computed)
temperature * time   <= 50000  ← inequality (violated suggestions projected to boundary)
```

> 📖 Constraint editor → [§ 3.10 Defining Parameter Constraints](DOCUMENTATION.md#310-defining-parameter-constraints)

---

## Context Variables (Uncontrollable Conditions)

Some measured quantities affect your experiment but cannot be controlled — ambient humidity, atmospheric pressure, reagent purity. Mark these as **context variables** so the model learns from them without suggesting values for them.

**The problem without context variables:**
> You optimise synthesis temperature. Unknown to the model, humidity spiked on several days, causing poor yields. The model wrongly concludes that *temperature* was the problem and avoids those conditions — discarding good parameter choices.

**With context variables:**
> The model learns `f(Temperature, Humidity) → Yield`. When you ask for the next batch on a humid day, it recommends temperatures that work *at that humidity level*.

### How it works

| | Controllable parameter | Context variable |
|---|---|---|
| You set it? | ✅ Yes | ❌ No — measured |
| Optimizer suggests values? | ✅ Yes | ❌ No |
| Surrogate learns from it? | ✅ Yes | ✅ Yes |
| Correctable after experiment? | Via cell editing | ✅ Yes — planned vs. actual |

**GUI:** In the Objectives panel, mark context columns in the "Context Variables" section, then fill in the **🌡 Current Conditions** panel before asking for each batch. If actual conditions differed (e.g. a humidity spike), correct the values in the Batch Results Dialog before submitting.

**Scripting:**
```python
from parameter_config import StudyConfig, ParameterConfig, ParameterType, ObjectiveConfig, ContextConfig
from optuna_builder import build_study, ask_batch, tell_batch

config = StudyConfig(
    parameters=[
        ParameterConfig(name="temperature", ptype=ParameterType.FLOAT,
                        full_min=50.0, full_max=200.0),
    ],
    objectives=[ObjectiveConfig(column_name="yield_pct", direction="maximize")],
    context_variables=[
        ContextConfig("humidity_pct", description="Ambient humidity (%)"),
    ],
    batch_size=3,
    sampler_name="TPE",
)

# Context values come from any source: database, sensor API, manual measurement
current_context = {"humidity_pct": fetch_from_sensor()}

trials = ask_batch(study, config, config.batch_size, context=current_context)

# After running experiments — correct if actual conditions differed
actual_context = [{"humidity_pct": 35.0}] * len(trials)   # e.g. humidity was higher
tell_batch(study, [t.number for t in trials], results,
           context_values=actual_context)
```

> 📖 Full guide → [§ 3.11 Context Variables](DOCUMENTATION.md#311-context-variables-uncontrollable-environmental-conditions)

---

## Power Analysis — Sample Size Calculator

Before committing to an experiment campaign, use the **🔬 Power** tab to answer:
> *"How many experiments N do I need to reliably detect an improvement of Δ, given that my measurements have a spread of σ?"*

The key relationship is **Cohen's d = Δ / σ** — the ratio of the effect you want to detect to the noise in your measurements:

| Cohen's d | Effect size | N needed (α=0.05, power=0.80) |
|---|---|---|
| 0.2 | Small | ~197 |
| 0.5 | Medium | ~32 |
| 0.8 | Large | ~13 |

Click **📊 From data** to auto-fill σ from your loaded objective column. The live plots update instantly as you adjust inputs.

> 📖 Full guide → [§ 3.12 Power Analysis](DOCUMENTATION.md#312-power-analysis--sample-size-calculator)

---

## Multi-Objective Optimisation

Select more than one result column in the Objectives panel. LabOpt switches automatically to NSGA-II and shows the Pareto front in the Results tab.

> 📖 Multi-objective setup → [§ 3.4 Setting Objectives](DOCUMENTATION.md#34-setting-objectives)

---

## Saving and Resuming Sessions

LabOpt auto-saves after every ask and every tell. Two files are created alongside your CSV:

```
labopt_study_20260822_232040.db            ← Optuna SQLite database
labopt_study_20260822_232040_session.json  ← Session metadata + pending batch
```

To resume: **File → Load Session** or **File → Recent Sessions**.

> 📖 Session management → [§ 3.8 Saving and Resuming Sessions](DOCUMENTATION.md#38-saving-and-resuming-sessions)

---

## Headless / Scripting Mode

Use the backend without any GUI:

```python
from parameter_config import StudyConfig, ParameterConfig, ParameterType, ObjectiveConfig
from optuna_builder import build_study, ask_batch, tell_batch, load_historical_trials
from session_manager import SessionManager

config = StudyConfig(
    parameters=[
        ParameterConfig(name="temperature", ptype=ParameterType.FLOAT,
                        full_min=50.0, full_max=200.0),
        ParameterConfig(name="time_min",    ptype=ParameterType.INT,
                        full_min=5,    full_max=120),
    ],
    objectives=[ObjectiveConfig(column_name="yield_pct", direction="maximize")],
    batch_size=4,
    n_batches=10,
    sampler_name="TPE",
)

state = SessionManager.create_new_session(config, csv_path="", session_dir="/tmp/my_run")
study = build_study(config, state.storage_path, state.study_name)

for _ in range(config.n_batches):
    trials = ask_batch(study, config, config.batch_size)
    results = [{"trial_number": t.number, "values": [my_measure(t.params)]}
               for t in trials]
    tell_batch(study, [r["trial_number"] for r in results],
               [r["values"] for r in results])
```

> 📖 Full scripting reference → [§ 4 Headless / Scripting Mode](DOCUMENTATION.md#4-headless--scripting-mode)

---

## Surrogate Validation

Before trusting the model's suggestions, verify that the surrogate is actually learning from your data. Tick **🔬 Surrogate Validation** in the Results tab (next to Export Report):

```
📊 Results tab  →  [📄 Export Report…]  [🔬 Surrogate Validation  ☐]
                   [████████████████░░░░]  45% — §2 BO Benchmark…
                   ✅  6/8 checks passed  (YELLOW)
```

The engine runs five sections:

| Section | What it checks |
|---|---|
| §0 Data Quality | Distribution of the objective; within-replicate variability |
| §1 Surrogate Accuracy | RF + GP hold-out Pearson r & Spearman ρ; CV RMSE < 30 % of range |
| §2 BO Benchmark | GP+EI / RF+EI / Greedy / Random strategies on a fixed pool — does BO beat random? |
| §3 EI Marginals | 1-D Expected Improvement sweep per parameter — where should you explore next? |
| §5 Pass/Fail | 8 dataset-agnostic checks with GREEN / YELLOW / RED overall rating |

**Requirements:**
- ≥ 30 unique rows in the loaded CSV.
- Runtime: 1–2 min (30 rows) to 6–10 min (200 rows), depending on dataset size.

**Results:**
- 7 new tabs appear in the Design Space dialog automatically.
- All 7 plots are embedded in the **📄 Export Report** HTML.

> 📖 Full guide → [§ 3.13 Surrogate Validation](DOCUMENTATION.md#313-surrogate-validation)

---

## Design Space Visualisation

Open **📊 Design Space…** in the Results tab for:

- **Pairplot / parallel coordinates / marginals** — where your data lives in parameter space, coloured by objective value.
- **Batch overlay** — suggested experiments shown as pink stars before the results dialog opens.
- **Correlation matrix** — which parameters drive each objective.
- **Importance chart** — Random Forest permutation importances.
- **Profiler** — adjust sliders and see the predicted objective update in real time.

> 📖 Design space guide → [§ 3.9 Design Space Visualisation](DOCUMENTATION.md#39-design-space-visualisation)

---

## Full Documentation

All details, module references, architecture diagrams, and troubleshooting are in **[DOCUMENTATION.md](DOCUMENTATION.md)**.

| Section | Topic |
|---|---|
| [§ 3](DOCUMENTATION.md#3-how-to-use-the-application-gui-walkthrough) | Complete GUI walkthrough |
| [§ 3.11](DOCUMENTATION.md#311-context-variables-uncontrollable-environmental-conditions) | Context variables — how the surrogate learns from uncontrollable conditions |
| [§ 3.12](DOCUMENTATION.md#312-power-analysis--sample-size-calculator) | Power Analysis — sample size calculator |
| [§ 4](DOCUMENTATION.md#4-headless--scripting-mode) | Headless / scripting API |
| [§ 5](DOCUMENTATION.md#5-codebase-architecture) | Architecture & module dependency diagram |
| [§ 6](DOCUMENTATION.md#6-module-reference) | Module-level API reference |
| [§ 9](DOCUMENTATION.md#9-common-adjustments--recipes) | Common recipes & adjustments |
| [§ 10](DOCUMENTATION.md#10-troubleshooting) | Troubleshooting |
| [§ 11](DOCUMENTATION.md#11-assumptions-limitations--when-to-trust-the-results) | Assumptions, limitations & when to trust the results |

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `optuna` | ≥ 3.6 | Bayesian optimisation engine |
| `pyside6` | ≥ 6.7 | GUI framework |
| `pandas` | ≥ 2.0 | CSV handling |
| `numpy` | ≥ 1.26 | Numerical operations |
| `matplotlib` | ≥ 3.8 | Design space plots |
| `sqlalchemy` | ≥ 2.0 | Optuna's SQLite backend |
| `scikit-learn` | ≥ 1.4 | RF surrogate & quality metrics |
| `scipy` | ≥ 1.11 | Statistical utilities |
| `openpyxl` | ≥ 3.1 | Excel (.xlsx / .xls) read/write |

---

## License

See [Credits.txt](Credits.txt) for third-party acknowledgements.
