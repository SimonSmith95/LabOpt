"""
Application Logger
==================
Sets up a rotating file logger for LabOpt.

Log file : {user_home}/.labopt/labopt.log
Max size : 5 MB per file, 2 backup copies → 15 MB on-disk cap.
           When the active log reaches 5 MB the logging library rotates it:
           labopt.log.1 becomes labopt.log.2, labopt.log becomes labopt.log.1,
           and a fresh labopt.log is started.  The oldest file (labopt.log.2)
           is deleted.  Old entries are never appended to — they simply rotate
           out and are deleted automatically.

Usage
-----
    from app_logger import get_logger
    log = get_logger(__name__)

    log.debug("Detailed diagnostic information")
    log.info("Session loaded: %s", path)
    log.warning("RF importance skipped — fewer than 15 rows")
    log.error("Worker thread raised: %s", exc)
    log.critical("Uncaught exception", exc_info=True)

Public helpers
--------------
configure_logger()   Call once at application startup (main.py does this).
get_logger(name)     Returns a child logger under the 'labopt' namespace.
log_path()           Returns the absolute path of the active log file.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# ── Log file location & rotation settings ────────────────────────────────────
_LOG_DIR     = os.path.join(os.path.expanduser("~"), ".labopt")
_LOG_FILE    = os.path.join(_LOG_DIR, "labopt.log")
_MAX_BYTES   = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 2                  # keep 2 rotated copies → 15 MB total cap

# ── Internal flag so configure_logger() is idempotent ───────────────────────
_configured = False


def configure_logger(*, console_level: int = logging.WARNING) -> None:
    """
    Configure the root 'labopt' logger.

    Call this exactly once, early in ``main.py``, before any module calls
    ``get_logger()``.  Subsequent calls are no-ops.

    Parameters
    ----------
    console_level : int
        Minimum severity shown on stdout.  Defaults to WARNING so normal runs
        are quiet on the terminal.  Pass ``logging.DEBUG`` for verbose output.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Ensure the log directory exists
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except OSError:
        pass   # best-effort; if it fails, the file handler will fail gracefully

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("labopt")
    root.setLevel(logging.DEBUG)

    # ── Rotating file handler (primary sink) ──────────────────────────────
    try:
        file_h = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,   # don't create the file until first write
        )
        file_h.setLevel(logging.DEBUG)
        file_h.setFormatter(fmt)
        root.addHandler(file_h)
    except Exception as exc:
        # Log directory may be read-only on some deployments — don't crash
        print(f"[LabOpt] WARNING: could not set up log file at {_LOG_FILE}: {exc}",
              file=sys.stderr)

    # ── Console handler (WARNING+ only) ───────────────────────────────────
    con_h = logging.StreamHandler(sys.stderr)
    con_h.setLevel(console_level)
    con_h.setFormatter(fmt)
    root.addHandler(con_h)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``labopt`` namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module, e.g.
        ``"labopt.main_window"`` is produced from ``get_logger(__name__)``.
    """
    if not _configured:
        configure_logger()
    # Strip the package prefix so callers can pass plain __name__
    if name.startswith("labopt."):
        child = name
    else:
        child = f"labopt.{name.rsplit('.', 1)[-1]}"
    return logging.getLogger(child)


def log_path() -> str:
    """Return the absolute path of the active (newest) log file."""
    return _LOG_FILE
