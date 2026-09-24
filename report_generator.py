"""
Feature 9 — Export Summary Report (HTML)
=========================================
Generates a fully self-contained HTML report of an optimisation session.

All images are embedded as base64-encoded PNGs — the file works offline
without any internet connection or CDN dependencies.

Public API
----------
fig_to_base64(fig) -> str
    Convert a Matplotlib Figure to a base64 PNG string.

generate_report(metadata, trials_df, figures, best_rows) -> str
    Build and return the complete HTML string.

write_report(html, directory) -> str
    Write *html* to a timestamped file in *directory*; return the path.
"""
from __future__ import annotations

import base64
import datetime
import io
import os
from typing import Dict, List, Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ──────────────────────────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    """
    Render *fig* (a Matplotlib Figure) to a PNG and return it base64-encoded.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    Returns
    -------
    str — base64-encoded PNG, suitable for embedding in an ``<img>`` src.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ──────────────────────────────────────────────────────────────────────────────
# Inline CSS / HTML helpers
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
    background: #1e1e2e;
    color: #cdd6f4;
    margin: 0;
    padding: 0;
}
.page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px 32px 48px;
}
h1 { color: #89dceb; border-bottom: 2px solid #45475a; padding-bottom: 8px; }
h2 { color: #89b4fa; border-bottom: 1px solid #313244; padding-bottom: 4px;
     margin-top: 36px; }
h3 { color: #cba6f7; margin-top: 20px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 20px;
    font-size: 12px;
}
th {
    background: #2a2a3e;
    color: #89b4fa;
    font-weight: bold;
    padding: 7px 10px;
    border: 1px solid #45475a;
    text-align: left;
}
td {
    padding: 6px 10px;
    border: 1px solid #313244;
}
tr:nth-child(even) td { background: #24243e; }
tr:hover td { background: #313244; }
.meta-grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 6px 20px;
    margin: 12px 0 24px;
    font-size: 13px;
}
.meta-key { color: #89b4fa; font-weight: bold; }
.meta-val { color: #cdd6f4; }
.fig-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin: 16px 0;
}
.fig-box {
    flex: 1 1 480px;
    min-width: 300px;
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 6px;
    overflow: hidden;
}
.fig-caption {
    background: #2a2a3e;
    color: #bac2de;
    font-size: 11px;
    padding: 5px 10px;
    text-align: center;
}
.fig-box img { width: 100%; display: block; }
.placeholder {
    color: #585b70;
    font-style: italic;
    margin: 10px 0;
}
.badge-good    { color: #a6e3a1; font-weight: bold; }
.badge-moderate{ color: #f9e2af; font-weight: bold; }
.badge-poor    { color: #f38ba8; font-weight: bold; }
footer {
    margin-top: 40px;
    border-top: 1px solid #313244;
    padding-top: 10px;
    color: #585b70;
    font-size: 11px;
    text-align: center;
}
"""


def _html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Render a DataFrame as an HTML table (max *max_rows* rows)."""
    if df.empty:
        return "<p class='placeholder'>No data available.</p>"
    truncated = len(df) > max_rows
    display_df = df.head(max_rows)
    rows = []
    for _, row in display_df.iterrows():
        cells = "".join(
            f"<td>{_fmt(v)}</td>" for v in row
        )
        rows.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{c}</th>" for c in display_df.columns)
    note = (
        f"<p class='placeholder'>Showing first {max_rows} of {len(df)} rows.</p>"
        if truncated else ""
    )
    return (
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{note}"
    )


def _fmt(val) -> str:
    """Format a table cell value for display."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


def _img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def _fig_section(label: str, b64: Optional[str]) -> str:
    if not b64:
        return ""
    return (
        f'<div class="fig-box">'
        f'<div class="fig-caption">{label}</div>'
        f'{_img_tag(b64, label)}'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    metadata: dict,
    trials_df: pd.DataFrame,
    figures: Dict[str, object],   # {label: matplotlib.figure.Figure}
    best_rows: List[dict],
) -> str:
    """
    Build a self-contained HTML report of an optimisation session.

    Parameters
    ----------
    metadata  : dict with keys:
                    session_name, date, sampler, n_batches, objectives
                    (all string-valued; objectives may be a list of str)
    trials_df : DataFrame of all completed trials (may be empty).
    figures   : Mapping of label → Matplotlib Figure.  Recognised keys:
                    "convergence", "pareto", "design_space", "correlation",
                    "importance"  (any others are included under their label).
    best_rows : List of dicts for the Pareto / best-trial table.

    Returns
    -------
    str — complete HTML document.
    """
    session_name = metadata.get("session_name", "LabOpt Session")
    date_str     = metadata.get("date", datetime.date.today().isoformat())
    sampler      = metadata.get("sampler", "—")
    n_batches    = metadata.get("n_batches", "—")
    objectives   = metadata.get("objectives", [])
    if isinstance(objectives, str):
        objectives = [objectives]
    obj_str = ", ".join(objectives) if objectives else "—"

    n_trials = len(trials_df) if not trials_df.empty else 0

    # Optional project identity fields (added by the DoE / session naming feature)
    project_name = metadata.get("project_name", "")
    owner        = metadata.get("owner", "")
    description  = metadata.get("description", "")
    doe_strategy = metadata.get("doe_strategy", "")  # e.g. "LHS (20 pts)"

    # ── Metadata block ─────────────────────────────────────────────────────
    meta_rows = ""
    if project_name:
        meta_rows += f'<span class="meta-key">Project</span><span class="meta-val">{project_name}</span>'
    if owner:
        meta_rows += f'<span class="meta-key">Researcher</span><span class="meta-val">{owner}</span>'
    if description:
        meta_rows += f'<span class="meta-key">Description</span><span class="meta-val">{description}</span>'
    meta_rows += (
        f'<span class="meta-key">Session</span><span class="meta-val">{session_name}</span>'
        f'<span class="meta-key">Date</span><span class="meta-val">{date_str}</span>'
        f'<span class="meta-key">Sampler</span><span class="meta-val">{sampler}</span>'
        f'<span class="meta-key">Batches completed</span><span class="meta-val">{n_batches}</span>'
        f'<span class="meta-key">Objectives</span><span class="meta-val">{obj_str}</span>'
        f'<span class="meta-key">Total trials</span><span class="meta-val">{n_trials}</span>'
    )
    if doe_strategy:
        meta_rows += f'<span class="meta-key">DoE strategy</span><span class="meta-val">{doe_strategy}</span>'
    meta_html = f'<div class="meta-grid">{meta_rows}</div>'

    # ── Best / Pareto table ────────────────────────────────────────────────
    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_html = _html_table(best_df)
    elif not trials_df.empty:
        best_html = "<p class='placeholder'>No best-trial data available.</p>"
    else:
        best_html = "<p class='placeholder'>No trials completed yet.</p>"

    # ── All trials table ───────────────────────────────────────────────────
    trials_html = _html_table(trials_df) if not trials_df.empty \
        else "<p class='placeholder'>No trials completed yet.</p>"

    # ── Figures ────────────────────────────────────────────────────────────
    # Convert each Figure → base64 PNG (or None if conversion fails)
    b64: Dict[str, Optional[str]] = {}
    _label_order = [
        ("convergence",  "Convergence — Best Value vs. Trial"),
        ("pareto",       "Pareto Front Scatter"),
        ("design_space", "Design Space Pairplot"),
        ("correlation",  "Correlation Matrix"),
        ("importance",   "Parameter Importance"),
    ]
    for key, _ in _label_order:
        fig = figures.get(key)
        if fig is not None:
            try:
                b64[key] = fig_to_base64(fig)
            except Exception:
                b64[key] = None
        else:
            b64[key] = None

    # Any extra figures not in the standard list (skip validation_* keys — handled separately)
    extra_figs: List[tuple[str, Optional[str]]] = []
    known_keys = {k for k, _ in _label_order}
    _val_prefix = "validation_"
    for key, fig in figures.items():
        if key not in known_keys and not key.startswith(_val_prefix) and fig is not None:
            try:
                extra_figs.append((key, fig_to_base64(fig)))
            except Exception:
                extra_figs.append((key, None))

    # Build figure rows (pair them side-by-side where possible)
    fig_rows_html = '<div class="fig-row">'
    any_fig = False
    for key, label in _label_order:
        if b64.get(key):
            fig_rows_html += _fig_section(label, b64[key])
            any_fig = True
    for label, img in extra_figs:
        if img:
            fig_rows_html += _fig_section(label, img)
            any_fig = True
    fig_rows_html += "</div>"
    if not any_fig:
        fig_rows_html = "<p class='placeholder'>No plots available (open the app and run batches to generate plots, then re-export).</p>"

    # ── Surrogate Validation section (optional) ────────────────────────────
    _val_section_titles = {
        "validation_s0_data_summary":        "§0  Data Quality",
        "validation_s1_predicted_vs_true":   "§1  Surrogate Accuracy (Predicted vs. True)",
        "validation_s1_residuals":           "§1  Residuals",
        "validation_s2_bo_convergence":      "§2  BO Benchmark — Convergence",
        "validation_s2_bo_convergence_norm": "§2  BO Benchmark — Normalised",
        "validation_s3_ei_marginals":        "§3  Expected Improvement Marginals",
        "validation_s5_passfail":            "§5  Validation Pass/Fail Summary",
    }

    val_html = ""
    if any(k.startswith(_val_prefix) for k in figures):
        val_html_parts = [
            "<h2>Surrogate Validation</h2>",
            "<p>The following plots were generated by the generalised surrogate "
            "validation suite (§0–§5).  They assess whether the model can reliably "
            "predict held-out data and whether Bayesian Optimisation outperforms "
            "random sampling on this dataset.</p>",
        ]
        for key, title in _val_section_titles.items():
            fig = figures.get(key)
            if fig is None:
                continue
            try:
                img_b64 = fig_to_base64(fig)
            except Exception:
                continue
            val_html_parts.append(f"<h3>{title}</h3>")
            val_html_parts.append(
                f'<div class="fig-box" style="flex-basis:100%;max-width:100%;">'
                f'<div class="fig-caption">{title}</div>'
                f'{_img_tag(img_b64, title)}'
                f'</div>'
            )
        val_html = "\n".join(val_html_parts)

    # ── Assemble HTML ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LabOpt Report — {session_name}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="page">

<h1>🔬 LabOpt Optimisation Report</h1>

<h2>Session Summary</h2>
{meta_html}

<h2>Best / Pareto Results</h2>
{best_html}

<h2>Plots</h2>
{fig_rows_html}

{val_html}

<h2>All Completed Trials</h2>
{trials_html}

<footer>
  Generated by LabOpt &mdash; {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  &nbsp;|&nbsp; File is fully self-contained (no internet connection required).
</footer>

</div>
</body>
</html>
"""
    return html


def doe_report(
    doe_state,          # DoEState | None
    doe_df,             # pd.DataFrame | None — generated points
    config,             # StudyConfig
    session_state=None, # SessionState | None — for project metadata
) -> str:
    """
    Generate a self-contained HTML report of the DoE phase.

    Sections
    --------
    §0  Session / project metadata
    §1  Design summary (strategy, N, readiness target)
    §2  Coverage metrics table
    §3  Pairplot (embedded as base64 PNG, if available)
    §4  1D marginal uniformity (embedded, if available)
    §5  Main effects (if results entered)
    §6  Full DoE table (all points + results)
    §7  Surrogate readiness status

    Parameters
    ----------
    doe_state     : DoEState with points, results, strategy, etc.
    doe_df        : DataFrame of generated DoE points.
    config        : StudyConfig (for parameter / objective information).
    session_state : Optional SessionState (for project_name, owner, etc.)

    Returns
    -------
    str — complete HTML document.
    """
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Helper: inline figure ──────────────────────────────────────────────
    def _fig_html(fig, caption: str) -> str:
        try:
            b64 = fig_to_base64(fig)
            plt.close(fig)
            return (
                f'<div class="fig-box" style="flex-basis:100%;max-width:900px;margin:12px auto">'
                f'<div class="fig-caption">{caption}</div>'
                f'{_img_tag(b64, caption)}'
                f'</div>'
            )
        except Exception:
            return f'<p class="placeholder">[{caption} could not be rendered]</p>'

    # ── Metadata ───────────────────────────────────────────────────────────
    project = ""
    owner   = ""
    desc    = ""
    if session_state is not None:
        project = getattr(session_state, "project_name", "")
        owner   = getattr(session_state, "owner", "")
        desc    = getattr(session_state, "description", "")

    strategy = doe_state.strategy if doe_state else "—"
    n_pts    = doe_state.n_points if doe_state else 0
    seed     = doe_state.seed if doe_state else 0
    n_results = sum(1 for r in (doe_state.results or []) if r is not None) if doe_state else 0

    # Readiness target
    try:
        from doe_math import compute_readiness_target
        rt = compute_readiness_target(config)
        target   = rt["target"]
        formula  = rt["formula_str"]
    except Exception:
        target, formula = "—", "—"

    meta_html = (
        '<div class="meta-grid">'
        + (f'<span class="meta-key">Project</span><span class="meta-val">{project}</span>' if project else "")
        + (f'<span class="meta-key">Researcher</span><span class="meta-val">{owner}</span>' if owner else "")
        + (f'<span class="meta-key">Description</span><span class="meta-val">{desc}</span>' if desc else "")
        + f'<span class="meta-key">Strategy</span><span class="meta-val">{strategy}</span>'
        + f'<span class="meta-key">N points</span><span class="meta-val">{n_pts}</span>'
        + f'<span class="meta-key">Seed</span><span class="meta-val">{seed}</span>'
        + f'<span class="meta-key">Results entered</span><span class="meta-val">{n_results} / {n_pts}</span>'
        + f'<span class="meta-key">Readiness target</span><span class="meta-val">{target} runs</span>'
        + f'<span class="meta-key">Readiness formula</span><span class="meta-val">{formula}</span>'
        + f'<span class="meta-key">Date</span><span class="meta-val">{datetime.date.today().isoformat()}</span>'
        + '</div>'
    )

    # ── Coverage metrics table ─────────────────────────────────────────────
    coverage_html = ""
    if doe_df is not None and not doe_df.empty:
        try:
            from doe_math import coverage_metrics
            m = coverage_metrics(doe_df)
            mm = m.get("maximin_distance")
            dr = m.get("discrepancy")
            ref_mm = m.get("random_maximin_ref")
            ref_dr = m.get("random_discr_ref")
            rows = []
            if mm is not None:
                pct = f"+{100*(mm-ref_mm)/max(ref_mm,1e-9):.0f}%" if ref_mm and mm > ref_mm else ""
                rows.append(f"<tr><td>Maximin distance</td><td>{mm:.4f} {pct}</td>"
                            f"<td>Random baseline: {ref_mm:.4f}</td></tr>")
            if dr is not None:
                pct = f"{100*(ref_dr-dr)/max(ref_dr,1e-9):.0f}% better" if ref_dr and dr < ref_dr else ""
                rows.append(f"<tr><td>L2 discrepancy</td><td>{dr:.4f} {pct}</td>"
                            f"<td>Random baseline: {ref_dr:.4f}</td></tr>")
            if rows:
                coverage_html = (
                    "<h2>Coverage Metrics</h2>"
                    "<table><thead><tr><th>Metric</th><th>DoE value</th><th>Reference</th></tr></thead>"
                    "<tbody>" + "".join(rows) + "</tbody></table>"
                )
        except Exception:
            coverage_html = ""

    # ── Pairplot ───────────────────────────────────────────────────────────
    pairplot_html = ""
    if doe_df is not None and not doe_df.empty:
        try:
            from doe_visualisations import (
                _BG, _FG, _BLUE, _GRID, _apply_dark_style
            )
            import numpy as np
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm

            nc = doe_df.select_dtypes(include=[np.number]).columns.tolist()[:6]
            d = len(nc)
            if d > 0:
                # Build colour if results exist
                colours = [_BLUE] * len(doe_df)
                has_r = doe_state and any(r is not None for r in doe_state.results)
                if has_r:
                    y_vals = [r[0] if (r is not None and len(r) > 0) else None
                              for r in doe_state.results]
                    nv = [v for v in y_vals if v is not None]
                    if nv:
                        norm = mcolors.Normalize(vmin=min(nv), vmax=max(nv))
                        colours = [cm.viridis(norm(v)) if v is not None else (0.4,0.4,0.4,0.5)
                                   for v in y_vals]
                fig_pp, axes = plt.subplots(d, d, figsize=(min(2*d, 12), min(2*d, 12)))
                if d == 1:
                    axes = np.array([[axes]])
                for i in range(d):
                    for j in range(d):
                        ax = axes[i, j] if d > 1 else axes[0, 0]
                        if i == j:
                            ax.hist(doe_df[nc[i]].values, bins=min(8, len(doe_df)),
                                    color=_BLUE, alpha=0.7, edgecolor=_BG)
                        else:
                            ax.scatter(doe_df[nc[j]].values, doe_df[nc[i]].values,
                                       c=colours, s=16, alpha=0.8, linewidths=0)
                        ax.set_facecolor(_BG)
                        ax.tick_params(colors=_FG, labelsize=5)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(_GRID)
                        if i == d - 1:
                            ax.set_xlabel(nc[j], color=_FG, fontsize=7)
                        else:
                            ax.set_xticklabels([])
                        if j == 0:
                            ax.set_ylabel(nc[i], color=_FG, fontsize=7)
                        else:
                            ax.set_yticklabels([])
                _apply_dark_style(fig_pp)
                fig_pp.tight_layout()
                pairplot_html = "<h2>Pairplot</h2>" + _fig_html(fig_pp, "DoE Pairplot")
        except Exception:
            pairplot_html = ""

    # ── DoE Table ──────────────────────────────────────────────────────────
    table_html = ""
    if doe_state is not None and doe_state.points:
        obj_names = [o.column_name for o in config.objectives]
        rows_data = []
        for i, pt in enumerate(doe_state.points):
            row = dict(pt)
            result = doe_state.results[i] if i < len(doe_state.results) else None
            for oi, ocol in enumerate(obj_names):
                if result is not None and oi < len(result):
                    row[ocol] = round(float(result[oi]), 6)
                else:
                    row[ocol] = ""
            rows_data.append(row)
        doe_table_df = pd.DataFrame(rows_data)
        table_html = "<h2>DoE Points Table</h2>" + _html_table(doe_table_df)

    # ── Readiness status ───────────────────────────────────────────────────
    if isinstance(target, int):
        pct_ready = min(100, int(100 * n_results / max(1, target)))
        if pct_ready >= 100:
            status_colour = "#2ecc71"
            status_text = "✅ Readiness target met — ready to hand off to BO."
        elif pct_ready >= 50:
            status_colour = "#f39c12"
            status_text = f"⚠ {pct_ready}% of readiness target reached."
        else:
            status_colour = "#e74c3c"
            status_text = f"🔴 {pct_ready}% of readiness target — more experiments recommended."
        readiness_html = (
            f'<div style="background:{status_colour};color:#1e1e2e;padding:10px 16px;'
            f'border-radius:6px;margin:12px 0;font-weight:bold">'
            f'Surrogate Readiness: {n_results} / {target} runs ({pct_ready}%)&nbsp;—&nbsp;{status_text}'
            f'</div>'
        )
    else:
        readiness_html = ""

    # ── Assemble ───────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LabOpt DoE Report — {project or 'Session'}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="page">

<h1>🧪 LabOpt DoE Phase Report</h1>

<h2>Session Metadata</h2>
{meta_html}

{readiness_html}

{coverage_html}

{pairplot_html}

{table_html}

<footer>
  Generated by LabOpt DoE Reporter &mdash; {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  &nbsp;|&nbsp; Self-contained — no internet connection required.
</footer>

</div>
</body>
</html>
"""
    return html


def write_report(html: str, directory: str) -> str:
    """
    Write *html* to a timestamped file inside *directory*.

    Parameters
    ----------
    html      : Complete HTML string (from generate_report).
    directory : Directory path where the file will be saved.

    Returns
    -------
    str — Absolute path to the written file.
    """
    os.makedirs(directory, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"labopt_report_{ts}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


# ── Friendly display names for the standard figure keys ──────────────────────
_FIGURE_NAMES: dict[str, str] = {
    "convergence":  "convergence",
    "pareto":       "pareto_front",
    "design_space": "design_space",
    "correlation":  "correlation_matrix",
    "importance":   "parameter_importance",
}


def export_plots_as_png(
    figures: Dict[str, object],
    report_path: str,
    dpi: int = 200,
) -> str:
    """
    Save each figure in *figures* as a high-resolution PNG.

    Files are written to a companion folder named after the HTML report:
    ``<report_stem>_plots/``.  For example, if *report_path* is
    ``/sessions/labopt_report_20260908_221901.html``, the folder will be
    ``/sessions/labopt_report_20260908_221901_plots/`` and will contain
    files such as ``convergence.png``, ``design_space.png``, etc.

    Parameters
    ----------
    figures     : Mapping of label → Matplotlib Figure (same dict passed to
                  ``generate_report()``).
    report_path : Absolute path to the already-written HTML report.  The PNG
                  folder is created alongside it.
    dpi         : Output resolution in dots per inch.  Default 200 is suitable
                  for PowerPoint / Word slides.

    Returns
    -------
    str — Absolute path to the PNG folder (created even if figures is empty).
    """
    stem       = os.path.splitext(os.path.basename(report_path))[0]
    plots_dir  = os.path.join(os.path.dirname(report_path), f"{stem}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for key, fig in figures.items():
        if fig is None:
            continue
        filename = _FIGURE_NAMES.get(key, key.replace(" ", "_")) + ".png"
        out_path = os.path.join(plots_dir, filename)
        try:
            fig.savefig(
                out_path,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
        except Exception:
            pass   # non-fatal — skip figures that fail to render

    return plots_dir
