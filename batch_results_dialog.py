"""
Phase 8 — Batch Results Dialog
After a batch is suggested, lets the user enter or import measured lab results
before they are told back to the Optuna study.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from parameter_config import ObjectiveConfig
from sampler_utils import find_near_duplicates
from session_manager import SessionManager, SessionState


class BatchResultsDialog(QDialog):
    """
    Modal dialog displayed after ask_batch() completes.

    Shows the suggested experiment parameters (read-only) and collects the
    measured objective values from the user.  Results can be imported
    automatically from an updated CSV or entered manually.

    Feature 11 — Outlier Detection
    --------------------------------
    When ≥5 completed trials exist, a Random Forest is fit on the historical
    data.  Each time the user edits an objective spinbox, the entered value is
    compared against the RF prediction.  If the residual exceeds 2.5 standard
    deviations of training residuals the spinbox turns orange and shows a
    tooltip warning.  The warning is advisory only — Submit All always works.

    Signals
    -------
    results_submitted(list)
        Emitted when the user clicks "Submit All".
        Payload: list of {"trial_number": int, "values": list[float]}
    """

    results_submitted = Signal(list)

    # ── Spinbox style constants ─────────────────────────────────────────────
    _SPIN_STYLE = (
        "QDoubleSpinBox {"
        "  background-color: #ffffff;"
        "  color: #1a1a2e;"
        "  border: 2px solid #89b4fa;"
        "  border-radius: 4px;"
        "  padding: 3px 6px;"
        "  font-size: 13px;"
        "}"
        "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }"
    )
    _OUTLIER_SPIN_STYLE = (
        "QDoubleSpinBox {"
        "  background-color: #ffa500;"
        "  color: #1a1a2e;"
        "  border: 2px solid #ff8c00;"
        "  border-radius: 4px;"
        "  padding: 3px 6px;"
        "  font-size: 13px;"
        "}"
        "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }"
    )
    _MIN_TRIALS_FOR_OUTLIER = 5

    def __init__(
        self,
        pending_batch: List[dict],          # [{"trial_number": int, "params": dict}, ...]
        objectives: List[ObjectiveConfig],
        session_state: SessionState,
        parent: QWidget | None = None,
        existing_trials: Optional[List[dict]] = None,   # Feature 4: near-duplicate detection
        study=None,                                      # optuna.Study — for Skip/purge
        predictions: Optional[List] = None,             # RF-predicted objective values
    ) -> None:
        super().__init__(parent)
        self._pending_batch = list(pending_batch)   # copy so we can mutate
        self._objectives = objectives
        self._session_state = session_state
        self._existing_trials = existing_trials
        self._study = study
        self._predictions = predictions             # list[float | None] or None
        # Set of trial numbers the user has chosen to skip (purge)
        self._skipped_trials: set[int] = set()
        # Near-duplicate hits (list of dicts from find_near_duplicates)
        self._dup_hits: list[dict] = []

        self.setWindowTitle("Enter Batch Results")
        self.setMinimumWidth(740)
        self.setMinimumHeight(420)
        # Never grow taller than the available screen — QTableWidget scrolls internally
        _screen = QApplication.primaryScreen()
        if _screen is not None:
            _avail = _screen.availableGeometry()
            self.resize(
                min(920, _avail.width()  - 60),
                min(680, _avail.height() - 80),
            )
            self.setMaximumSize(_avail.width() - 40, _avail.height() - 40)
        else:
            self.resize(920, 680)
        self._build_ui()

        # ── Feature 11: RF-based outlier detection ─────────────────────────
        # Fit a surrogate on completed trials so we can flag implausible values.
        # _param_names / _obj_spins are set by _build_ui(); must come AFTER it.
        self._rf = None           # RandomForestRegressor | None
        self._pred_std = 0.0      # std of training residuals
        self._param_encodings: dict = {}   # param_name → {str_val: int_code}
        self._fit_outlier_rf()
        self._connect_outlier_signals()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        n = len(self._pending_batch)
        layout.addWidget(
            QLabel(f"<b>Enter measured results for {n} suggested experiment(s).</b>")
        )
        layout.addWidget(
            QLabel(
                "Parameter columns show the <b>suggested</b> target values.  "
                "You can <b>edit them</b> to enter the <i>actual</i> composition "
                "you achieved in the lab before submitting."
            )
        )

        # ── Toolbar buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        export_btn = QPushButton("⬆  Export Suggested Params as CSV")
        export_btn.setToolTip(
            "Save the suggested parameter values to a CSV file for use in the lab."
        )
        export_btn.clicked.connect(self._export_params_csv)
        btn_row.addWidget(export_btn)

        import_btn = QPushButton("⬇  Import Results from CSV")
        import_btn.setToolTip(
            "Load an updated CSV and auto-fill matched objective cells."
        )
        import_btn.clicked.connect(self._import_results_csv)
        btn_row.addWidget(import_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Pre-compute near-duplicate hits ───────────────────────────────
        if self._existing_trials and self._session_state.study_config:
            try:
                suggestions = [item["params"] for item in self._pending_batch]
                self._dup_hits = find_near_duplicates(
                    suggestions,
                    self._existing_trials,
                    self._session_state.study_config,
                )
            except Exception:
                self._dup_hits = []

        # Index near-duplicate hits by suggestion index for quick lookup
        _dup_by_idx = {h["suggestion_idx"]: h for h in self._dup_hits}

        # ── Results table ──────────────────────────────────────────────────
        param_names: List[str] = (
            list(self._pending_batch[0]["params"].keys()) if self._pending_batch else []
        )
        obj_names = [o.column_name for o in self._objectives]
        # "Predicted" column(s) for the RF surrogate estimate — shown before Action
        _has_preds = bool(self._predictions)
        _pred_col_labels = (
            [f"Predicted {obj_names[0]}"] if _has_preds and obj_names else []
        )
        # "Action" column (Skip button) is always the last column
        all_cols = ["Trial #"] + param_names + obj_names + _pred_col_labels + ["Action"]

        self._table = QTableWidget(len(self._pending_batch), len(all_cols))
        self._table.setHorizontalHeaderLabels(all_cols)

        hdr = self._table.horizontalHeader()
        for i in range(len(all_cols)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        # Don't stretch the Action column — keep it compact
        hdr.setSectionResizeMode(len(all_cols) - 1, QHeaderView.Fixed)
        self._table.setColumnWidth(len(all_cols) - 1, 140)
        hdr.setStretchLastSection(False)
        # Allow double-click / key editing — param cells are editable;
        # trial # and objective spinboxes handle their own interaction.
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.AnyKeyPressed
        )

        _PARAM_TIP = (
            "Suggested target value — double-click to edit with the actual "
            "composition you achieved in the lab."
        )

        # Store spinboxes: self._obj_spins[row][obj_idx]
        self._obj_spins: List[List[QDoubleSpinBox]] = []
        self._param_names = param_names
        self._obj_names = obj_names

        for row_idx, trial_data in enumerate(self._pending_batch):
            col = 0
            trial_num = trial_data["trial_number"]
            is_dup = row_idx in _dup_by_idx

            # Trial number (always read-only)
            self._set_readonly_cell(row_idx, col, str(trial_num))
            col += 1

            # Param values — editable, amber tint to indicate they can be changed
            for pname in param_names:
                raw_val = trial_data["params"].get(pname, "")
                try:
                    display_val = f"{float(raw_val):.6g}"
                except (ValueError, TypeError):
                    display_val = str(raw_val)
                item = QTableWidgetItem(display_val)
                # Slightly different colour for near-duplicate rows
                bg_color = "#f59e0b" if not is_dup else "#c9850a"
                item.setBackground(QColor(bg_color))
                item.setForeground(QColor("#1a1a2e"))
                item.setToolTip(_PARAM_TIP)
                self._table.setItem(row_idx, col, item)
                col += 1

        # Objective spinboxes
            _SPIN_STYLE = (
                "QDoubleSpinBox {"
                "  background-color: #ffffff;"
                "  color: #1a1a2e;"
                "  border: 2px solid #89b4fa;"
                "  border-radius: 4px;"
                "  padding: 6px 8px;"
                "  font-size: 13px;"
                "  min-height: 32px;"
                "}"
                "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }"
            )
            row_spins: List[QDoubleSpinBox] = []
            for _obj in obj_names:
                spin = QDoubleSpinBox()
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(6)
                spin.setSingleStep(0.001)
                spin.setStyleSheet(_SPIN_STYLE)
                spin.setSpecialValueText(" ")
                spin.setValue(spin.minimum())
                spin.setMinimumHeight(34)
                self._table.setCellWidget(row_idx, col, spin)
                row_spins.append(spin)
                col += 1
            self._obj_spins.append(row_spins)

            # ── Surrogate-predicted value (read-only, italic) ──────────────
            if _has_preds:
                pred = (
                    self._predictions[row_idx]
                    if self._predictions and row_idx < len(self._predictions)
                    else None
                )
                if pred is not None:
                    pred_text = f"{pred:.4g}"
                    pred_tip  = (
                        f"RF surrogate estimate: {pred_text}\n"
                        "This is the model's prediction for the objective value\n"
                        "at this composition.  Compare with your actual result\n"
                        "to track model accuracy over time."
                    )
                else:
                    pred_text = "—"
                    pred_tip  = "Insufficient data for surrogate prediction."
                pred_item = QTableWidgetItem(pred_text)
                pred_item.setFlags(pred_item.flags() & ~Qt.ItemIsEditable)
                pred_item.setForeground(QColor("#89b4fa"))   # soft blue — read-only
                pred_item.setToolTip(pred_tip)
                font = pred_item.font()
                font.setItalic(True)
                pred_item.setFont(font)
                self._table.setItem(row_idx, col, pred_item)
                col += 1

            # ── Skip (Purge) button ────────────────────────────────────────
            skip_label = "⚠  Skip (Duplicate)" if is_dup else "Skip Trial"
            skip_btn = QPushButton(skip_label)
            if is_dup:
                hit = _dup_by_idx[row_idx]
                skip_btn.setToolTip(
                    f"{hit['message']}\n\n"
                    "Click to skip this trial. It will be marked as FAILED in Optuna\n"
                    "so the optimizer can suggest a different point next time."
                )
                skip_btn.setStyleSheet(
                    "QPushButton { background:#b45309; color:white; "
                    "border-radius:4px; padding:3px 6px; font-weight:bold; }"
                    "QPushButton:hover { background:#d97706; }"
                )
            else:
                skip_btn.setToolTip(
                    "Skip this trial — mark it as FAILED in Optuna so the optimizer\n"
                    "can suggest a different point in the next batch."
                )
                skip_btn.setStyleSheet(
                    "QPushButton { background:#374151; color:#9ca3af; "
                    "border-radius:4px; padding:3px 6px; }"
                    "QPushButton:hover { background:#4b5563; color:white; }"
                )
            skip_btn.clicked.connect(
                lambda _checked, r=row_idx, tn=trial_num: self._skip_trial(r, tn)
            )
            self._table.setCellWidget(row_idx, col, skip_btn)

        # ── Increase row height for readability ───────────────────────────
        self._table.verticalHeader().setDefaultSectionSize(44)
        self._table.resizeRowsToContents()

        layout.addWidget(self._table)

        # ── Near-duplicate warning banner (Feature 4) ──────────────────────
        if self._dup_hits:
            banner = QLabel(
                "⚠  Near-Duplicate Warning — some suggestions are very close to "
                "existing trials:<br>"
                + "<br>".join(f"• {h['message']}" for h in self._dup_hits)
                + "<br><i>Use the 'Skip' button in each row to purge that trial "
                "and ask Optuna for a different suggestion next time.</i>"
            )
            banner.setWordWrap(True)
            banner.setTextFormat(Qt.RichText)
            banner.setStyleSheet(
                "background: #3d2e00; color: #ffc107; "
                "padding: 8px; border-radius: 4px; "
                "border: 1px solid #ffc107;"
            )
            layout.addWidget(banner)

        # ── Dialog buttons ─────────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        submit_btn = btn_box.addButton("Submit All", QDialogButtonBox.AcceptRole)
        submit_btn.clicked.connect(self._on_submit)
        cancel_btn = btn_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(btn_box)

    def _set_readonly_cell(self, row: int, col: int, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, col, item)
        return item

    # ── Slot handlers ──────────────────────────────────────────────────────

    def _skip_trial(self, row_idx: int, trial_number: int) -> None:
        """
        Purge a trial the user does not want to run.

        1. Marks the row grey in the table (disabled).
        2. Adds the trial to ``self._skipped_trials`` so it is excluded on submit.
        3. If a study was provided, immediately tells Optuna the trial FAILED,
           preventing it from remaining as a WAITING trial that blocks future asks.
        """
        # Confirm only for non-duplicate rows (duplicates have an obvious reason)
        is_dup = any(h["suggestion_idx"] == row_idx for h in self._dup_hits)
        if not is_dup:
            reply = QMessageBox.question(
                self,
                "Skip Trial",
                f"Mark Trial #{trial_number} as skipped (FAILED)?\n\n"
                "The optimizer will treat it as a failed experiment and\n"
                "suggest a different point in the next batch.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._skipped_trials.add(trial_number)

        # Grey out the entire row
        n_cols = self._table.columnCount()
        _GREY_BG = QColor("#2d2d3f")
        _GREY_FG = QColor("#6c7086")
        for c in range(n_cols):
            item = self._table.item(row_idx, c)
            if item is not None:
                item.setBackground(_GREY_BG)
                item.setForeground(_GREY_FG)
            widget = self._table.cellWidget(row_idx, c)
            if widget is not None:
                widget.setEnabled(False)
                widget.setToolTip("This trial has been skipped.")

        # Replace the Skip button with a "Skipped" label
        skipped_lbl = QPushButton("✓ Skipped")
        skipped_lbl.setEnabled(False)
        skipped_lbl.setStyleSheet(
            "QPushButton { background:#1e3a2f; color:#4ade80; "
            "border-radius:4px; padding:3px 6px; }"
        )
        action_col = self._table.columnCount() - 1
        self._table.setCellWidget(row_idx, action_col, skipped_lbl)

        # Tell Optuna the trial failed immediately (non-fatal)
        if self._study is not None:
            try:
                from optuna.trial import TrialState
                self._study.tell(trial_number, state=TrialState.FAIL)
            except Exception:
                pass   # non-fatal — Optuna might already have transitioned it

    def _on_submit(self) -> None:
        results: List[dict] = []

        for row_idx, row_spins in enumerate(self._obj_spins):
            trial_number = self._pending_batch[row_idx]["trial_number"]

            # Skip rows the user has purged
            if trial_number in self._skipped_trials:
                continue

            # ── Objective values ───────────────────────────────────────────
            values: List[float] = []
            for obj_idx, spin in enumerate(row_spins):
                if spin.value() == spin.minimum():
                    QMessageBox.warning(
                        self,
                        "Missing Values",
                        f"Row {row_idx + 1}, objective '{self._obj_names[obj_idx]}' "
                        "has no value.  Please fill in all cells before submitting.",
                    )
                    return
                values.append(spin.value())

            # ── Read back actual param values (user may have edited them) ──
            actual_params: dict = {}
            nominal_params: dict = self._pending_batch[row_idx]["params"]
            for pi, pname in enumerate(self._param_names):
                col_idx = 1 + pi       # col 0 = Trial #
                item = self._table.item(row_idx, col_idx)
                raw = item.text().strip() if item else ""
                if raw == "":
                    actual_params[pname] = nominal_params.get(pname)
                else:
                    try:
                        nom = nominal_params.get(pname)
                        if isinstance(nom, int):
                            actual_params[pname] = int(round(float(raw)))
                        else:
                            actual_params[pname] = float(raw)
                    except ValueError:
                        actual_params[pname] = raw

            results.append({
                "trial_number":   trial_number,
                "values":         values,
                "actual_params":  actual_params,
                "nominal_params": nominal_params,
                "_row_display":   row_idx + 1,  # 1-based for user messages
                "_pending_idx":   row_idx,       # for looking up self._predictions
            })

        # If all rows were skipped, inform the user and close
        if not results and self._skipped_trials:
            QMessageBox.information(
                self,
                "All Trials Skipped",
                f"All {len(self._skipped_trials)} trial(s) were skipped.\n"
                "Optuna has marked them as FAILED.\n"
                "Ask the next batch to get fresh suggestions.",
            )
            self.accept()
            return

        # ── Pre-submit validation ──────────────────────────────────────────
        # Run three checks and prompt for confirmation if any fail.
        # The user can still choose to submit (it's advisory, not blocking).
        issues = self._pre_submit_checks(results)
        if issues:
            msg = "\n\n".join(issues)
            reply = QMessageBox.question(
                self,
                "⚠  Submission Warning — Please Confirm",
                f"The following issues were detected before submitting:\n\n"
                f"{msg}\n\n"
                "Submit anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,   # default = "Back / Re-enter"
            )
            if reply != QMessageBox.Yes:
                return   # return to the form without submitting

        # Strip internal helper keys before emitting
        for r in results:
            r.pop("_row_display", None)
            r.pop("_pending_idx", None)

        self.results_submitted.emit(results)
        self.accept()

    def _pre_submit_checks(self, results: List[dict]) -> List[str]:
        """
        Run three validation checks across all pending results.

        Returns a list of human-readable warning strings.  Empty list = all OK.

        Checks
        ------
        1. Constraint violations — actual entered params vs study constraints.
        2. Large param deviations — suggested vs actual (> 20 % of param range).
        3. Outlier result value — entered objective vs RF OOB prediction (> 2.5σ).
        """
        issues: List[str] = []
        cfg = (
            self._session_state.study_config
            if self._session_state and self._session_state.study_config
            else None
        )

        for result in results:
            trial_num   = result["trial_number"]
            row_display = result.get("_row_display", "?")
            actual_params  = result["actual_params"]
            nominal_params = result["nominal_params"]
            values         = result["values"]
            prefix = f"Row {row_display} (Trial #{trial_num})"

            # ── Check 1: Constraint violations ────────────────────────────
            if cfg and cfg.constraints:
                for c in cfg.constraints:
                    try:
                        satisfied, violation = c.is_satisfied(actual_params)
                        if not satisfied:
                            # Build a readable summary of the actual expression value
                            try:
                                from parameter_config import ParameterConstraint
                                actual_val = ParameterConstraint.eval_expr(
                                    c.expression, actual_params
                                )
                                val_str = f"{actual_val:.4g}"
                            except Exception:
                                val_str = "?"
                            issues.append(
                                f"⚠ {prefix} — Constraint '{c.name}' violated:\n"
                                f"   {c.expression} = {val_str}  "
                                f"(expected {c.operator} {c.target})"
                            )
                    except Exception:
                        pass

            # ── Check 2: Large param deviations (suggested vs actual) ─────
            if cfg and cfg.parameters:
                for p in cfg.parameters:
                    if not p.enabled or p.name not in actual_params:
                        continue
                    if p.name not in nominal_params:
                        continue
                    try:
                        from parameter_config import ParameterType
                        if p.ptype not in (ParameterType.FLOAT, ParameterType.INT):
                            continue   # skip categorical / bool
                        actual_val  = float(actual_params[p.name])
                        nominal_val = float(nominal_params[p.name])
                        full_range  = max(1e-12, p.full_max - p.full_min)
                        deviation   = abs(actual_val - nominal_val) / full_range
                        if deviation > 0.20:   # more than 20 % of param range
                            issues.append(
                                f"⚠ {prefix} — '{p.name}' was suggested as "
                                f"{nominal_val:.4g} but you entered {actual_val:.4g} "
                                f"(\u0394 = {deviation * 100:.0f} % of range). "
                                "Was this intentional?"
                            )
                    except (TypeError, ValueError, AttributeError):
                        pass

            # ── Check 3: Outlier result vs reference prediction ─────────────
            # Uses the same dual-threshold as _check_outlier():
            #   - z-score > 2.5 σ  OR  relative error > 90 %
            # Priority: displayed prediction first, RF fallback second.
            if values:
                entered = values[0]   # first objective
                pending_idx = result.get("_pending_idx")
                predicted: float | None = None

                # Priority 1: the value already shown in the "Predicted X" column
                if (
                    self._predictions
                    and pending_idx is not None
                    and pending_idx < len(self._predictions)
                    and self._predictions[pending_idx] is not None
                ):
                    try:
                        predicted = float(self._predictions[pending_idx])
                    except (TypeError, ValueError):
                        predicted = None

                # Priority 2: OOB-RF prediction using actual entered params
                if predicted is None and self._rf is not None:
                    feature_vec: list[float] = []
                    for pname in self._param_names:
                        raw = actual_params.get(pname, nominal_params.get(pname, 0))
                        if isinstance(raw, str):
                            enc = self._param_encodings.get(pname, {})
                            feature_vec.append(float(enc.get(raw, 0)))
                        elif isinstance(raw, bool):
                            feature_vec.append(1.0 if raw else 0.0)
                        else:
                            try:
                                feature_vec.append(float(raw))
                            except (TypeError, ValueError):
                                feature_vec.append(0.0)
                    try:
                        predicted = float(self._rf.predict([feature_vec])[0])
                    except Exception:
                        predicted = None

                if predicted is not None:
                    residual       = abs(entered - predicted)
                    relative_error = residual / (abs(predicted) + 1.0)
                    z_outlier = self._pred_std > 1e-12 and residual > 2.5 * self._pred_std
                    r_outlier = relative_error > 0.90
                    if z_outlier or r_outlier:
                        obj_name = self._obj_names[0] if self._obj_names else "objective"
                        pct = relative_error * 100
                        issues.append(
                            f"⚠ {prefix} — '{obj_name}' entered ({entered:.4g}) "
                            f"is {pct:.0f}% away from the model's prediction "
                            f"({predicted:.4g}). Please double-check for typos."
                        )

        return issues

    def _on_cancel(self) -> None:
        reply = QMessageBox.question(
            self,
            "Cancel",
            "Cancel result entry?\n\n"
            "The pending batch is still saved in the session file.\n"
            "Use the '📋 Enter Pending Results…' toolbar button to reopen\n"
            "this dialog at any time — no need to reload the app.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.reject()

    # ── Export / Import ────────────────────────────────────────────────────

    def _export_params_csv(self) -> None:
        if not self._pending_batch:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Suggested Parameters", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        rows = [
            {"trial_number": td["trial_number"], **td["params"]}
            for td in self._pending_batch
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Suggested parameters saved to:\n{path}")

    def _import_results_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Results CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", f"Cannot read file:\n{exc}")
            return

        matches = SessionManager.match_pending_to_csv(self._session_state, df)
        if not matches:
            QMessageBox.information(
                self,
                "No Matches Found",
                "No rows in the CSV matched the pending trial parameters.\n\n"
                "Make sure the parameter column values are identical to those "
                "that were suggested.",
            )
            return

        self.prefill_results(matches)
        filled = len(matches)
        total = len(self._pending_batch)
        QMessageBox.information(
            self,
            "Import Complete",
            f"Pre-filled {filled} of {total} trial(s).\n"
            + (
                f"{total - filled} trial(s) were not matched and must be entered manually."
                if filled < total
                else "All trials matched!"
            ),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def prefill_results(self, matches: Dict[int, List[float]]) -> None:
        """
        Pre-fill objective spinboxes from a {trial_number: [values]} dict.
        Matched cells are highlighted green.
        """
        n_params = len(self._param_names)
        for row_idx, trial_data in enumerate(self._pending_batch):
            trial_num = trial_data["trial_number"]
            if trial_num not in matches:
                continue
            values = matches[trial_num]
            for obj_idx, val in enumerate(values):
                if obj_idx < len(self._obj_spins[row_idx]):
                    spin = self._obj_spins[row_idx][obj_idx]
                    spin.setValue(val)
                    # Visual highlight
                    spin.setStyleSheet("background-color: #d4edda;")  # light green

    # ── Feature 11 — Outlier detection ─────────────────────────────────────

    def _fit_outlier_rf(self) -> None:
        """
        Fit a Random Forest regressor on completed trials (first objective only).

        Requires ≥ ``_MIN_TRIALS_FOR_OUTLIER`` complete trials and sklearn.
        On any failure the method sets ``self._rf = None`` and returns silently —
        the dialog still works normally, just without outlier highlighting.
        """
        if self._study is None or not self._param_names or not self._objectives:
            return
        try:
            from optuna.trial import TrialState
            completed = [t for t in self._study.trials if t.state == TrialState.COMPLETE]
            if len(completed) < self._MIN_TRIALS_FOR_OUTLIER:
                return

            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            X_rows: list[list[float]] = []
            y_vals: list[float] = []

            for t in completed:
                # Use the first objective value only
                obj_val: float | None = None
                if t.values is not None and len(t.values) > 0:
                    obj_val = float(t.values[0])
                elif t.value is not None:
                    obj_val = float(t.value)
                if obj_val is None:
                    continue

                feature_row: list[float] = []
                for pname in self._param_names:
                    raw = t.params.get(pname, 0)
                    if isinstance(raw, str):
                        if pname not in self._param_encodings:
                            self._param_encodings[pname] = {}
                        enc = self._param_encodings[pname]
                        if raw not in enc:
                            enc[raw] = len(enc)
                        feature_row.append(float(enc[raw]))
                    elif isinstance(raw, bool):
                        feature_row.append(1.0 if raw else 0.0)
                    else:
                        try:
                            feature_row.append(float(raw))
                        except (TypeError, ValueError):
                            feature_row.append(0.0)
                X_rows.append(feature_row)
                y_vals.append(obj_val)

            if len(y_vals) < self._MIN_TRIALS_FOR_OUTLIER:
                return

            X = np.array(X_rows)
            y = np.array(y_vals)

            # Train with oob_score=True (bootstrap=True is sklearn default).
            # OOB predictions are out-of-bag — unbiased because each tree's
            # OOB samples were NOT used during that tree's training.
            # In-sample residuals would be near-zero (RF overfits training data),
            # making pred_std ≈ 0 and disabling the outlier flag entirely.
            rf = RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=1, oob_score=True
            )
            rf.fit(X, y)

            # oob_prediction_ has shape (n_samples,) — same indexing as y
            try:
                oob_preds = rf.oob_prediction_
                residuals = y - oob_preds
            except AttributeError:
                # Fallback to in-sample residuals (less accurate, but always available)
                residuals = y - rf.predict(X)

            std = float(np.std(residuals))
            if std < 1e-12:
                return   # degenerate: cannot compute meaningful z-score

            self._rf = rf
            self._pred_std = std

        except Exception:
            self._rf = None    # non-fatal — outlier detection simply disabled

    def _connect_outlier_signals(self) -> None:
        """Connect each objective spinbox's valueChanged to _check_outlier."""
        for row_idx, row_spins in enumerate(self._obj_spins):
            for obj_idx, spin in enumerate(row_spins):
                spin.valueChanged.connect(
                    # Use default-arg capture to avoid closure-over-loop-variable bug
                    lambda _val, r=row_idx, oi=obj_idx: self._check_outlier(r, oi)
                )

    def _check_outlier(self, row_idx: int, obj_idx: int) -> None:
        """
        Compare the spinbox value against the prediction and flag outliers.

        Priority order for the reference prediction
        -------------------------------------------
        1. ``self._predictions[row_idx]`` — the value already shown in the
           "Predicted …" column (same number the user can see).
        2. Internal OOB-RF prediction — used when predictions are unavailable.

        Flagging criteria (either triggers the orange highlight)
        --------------------------------------------------------
        - z-score  > 2.5 σ   (OOB-residual threshold)
        - relative error > 90 %  of the predicted value
          i.e. |entered − predicted| / (|predicted| + 1) > 0.90

        The relative-error criterion reliably catches "entered 5 vs predicted
        337,000" even when OOB residuals are large (noisy dataset).

        This method is deliberately silent on any error.
        """
        if obj_idx != 0:
            return   # outlier detection only for the first objective
        if row_idx >= len(self._obj_spins):
            return

        spin = self._obj_spins[row_idx][obj_idx]
        if spin.value() == spin.minimum():
            spin.setStyleSheet(self._SPIN_STYLE)
            spin.setToolTip("")
            return   # special "empty" sentinel — ignore

        entered = spin.value()

        # ── Resolve predicted value ────────────────────────────────────────
        predicted: float | None = None

        # Priority 1: the displayed prediction (same as "Predicted X" column)
        if (
            self._predictions
            and row_idx < len(self._predictions)
            and self._predictions[row_idx] is not None
        ):
            try:
                predicted = float(self._predictions[row_idx])
            except (TypeError, ValueError):
                predicted = None

        # Priority 2: internal OOB-RF prediction
        if predicted is None and self._rf is not None:
            feature_vec: list[float] = []
            for pname in self._param_names:
                raw = self._pending_batch[row_idx]["params"].get(pname, 0)
                if isinstance(raw, str):
                    enc = self._param_encodings.get(pname, {})
                    feature_vec.append(float(enc.get(raw, 0)))
                elif isinstance(raw, bool):
                    feature_vec.append(1.0 if raw else 0.0)
                else:
                    try:
                        feature_vec.append(float(raw))
                    except (TypeError, ValueError):
                        feature_vec.append(0.0)
            try:
                predicted = float(self._rf.predict([feature_vec])[0])
            except Exception:
                predicted = None

        if predicted is None:
            return   # no prediction available

        # ── Evaluate outlier criteria ──────────────────────────────────────
        residual       = abs(entered - predicted)
        relative_error = residual / (abs(predicted) + 1.0)

        z_outlier = self._pred_std > 1e-12 and residual > 2.5 * self._pred_std
        r_outlier = relative_error > 0.90   # more than 90 % away from predicted

        if z_outlier or r_outlier:
            pct = relative_error * 100
            spin.setStyleSheet(self._OUTLIER_SPIN_STYLE)
            spin.setToolTip(
                f"⚠ Model prediction: {predicted:.4g}. "
                f"Your entry ({entered:.4g}) is {pct:.0f}% away from the predicted value.\n"
                "This value is much further from the model's prediction than expected. "
                "Please double-check before submitting."
            )
        else:
            spin.setStyleSheet(self._SPIN_STYLE)
            spin.setToolTip("")
