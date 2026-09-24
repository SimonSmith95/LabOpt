"""
validation_worker.py
====================
PySide6 QThread wrapper around ValidationEngine.
Keeps the GUI responsive during the 3–10 minute validation run.

Public API
----------
ValidationWorker(X, y, feature_names, target_name, direction, ...)
    Signals:
        progress_updated(int, str)   — (percent, message)
        validation_done(object)      — ValidationResults
        error_occurred(str)          — traceback string
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from validate_generic import ValidationEngine, ValidationResults


class ValidationWorker(QThread):
    """
    Runs ValidationEngine in a background thread so the GUI stays responsive.

    Signals
    -------
    progress_updated : (int percent, str message)
        Emitted after each validation step as the engine progresses 0→100 %.
    validation_done  : (ValidationResults)
        Emitted once when the full run completes successfully.
    error_occurred   : (str traceback)
        Emitted if an unhandled exception propagates out of run().
    """

    progress_updated = Signal(int, str)       # (percent, message)
    validation_done  = Signal(object)         # ValidationResults
    error_occurred   = Signal(str)            # traceback

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_name: str,
        direction: str = "minimize",
        context_X: Optional[np.ndarray] = None,
        context_names: Optional[List[str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = ValidationEngine(
            X=X,
            y=y,
            feature_names=feature_names,
            target_name=target_name,
            direction=direction,
            context_X=context_X,
            context_names=context_names,
            progress_cb=self._on_progress,
        )

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress_updated.emit(pct, msg)

    def run(self) -> None:
        try:
            results: ValidationResults = self._engine.run()
            self.validation_done.emit(results)
        except Exception:
            import traceback
            self.error_occurred.emit(traceback.format_exc())
