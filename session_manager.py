"""
Phase 3 — Session Manager
Persists the full application state (StudyConfig + pending batch) to a JSON
file alongside the Optuna SQLite database, enabling close-and-resume workflow.

Extended in the DoE / Multi-user phase:
  - project_name, owner, description fields on SessionState
  - doe_state field for DoE phase persistence
  - LABOPT_SESSION_DIR / LABOPT_USER environment variable support
  - Per-owner subdirectory isolation
  - Volume-local registry when LABOPT_SESSION_DIR is set
  - list_all_sessions() for the Session Browser dialog
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from parameter_config import DoEState, StudyConfig

# ── Environment variable configuration ────────────────────────────────────────
_ENV_SESSION_DIR: str = os.environ.get("LABOPT_SESSION_DIR", "")
_ENV_USER: str = os.environ.get("LABOPT_USER", "")


def get_default_session_dir() -> str:
    """
    Returns the root directory for all sessions.
    Priority: LABOPT_SESSION_DIR env var > ~/labopt_sessions
    """
    if _ENV_SESSION_DIR:
        return _ENV_SESSION_DIR
    return os.path.join(os.path.expanduser("~"), "labopt_sessions")


def get_default_owner() -> str:
    """
    Returns the default owner string.
    Priority: LABOPT_USER env var > os.getlogin() > ""
    """
    if _ENV_USER:
        return _ENV_USER
    try:
        return os.getlogin()
    except Exception:
        return ""


def _get_registry_path() -> str:
    """
    Return the path to the global sessions registry JSON.
    • LABOPT_SESSION_DIR set → registry lives on the persistent volume
    • Otherwise            → user home directory (unchanged for local installs)
    """
    if _ENV_SESSION_DIR:
        return os.path.join(_ENV_SESSION_DIR, ".labopt_registry.json")
    return os.path.join(os.path.expanduser("~"), ".labopt_sessions.json")


# Keep the legacy constant for code that still references it directly.
RECENT_SESSIONS_PATH = _get_registry_path()


def _slugify(text: str, max_len: int = 40) -> str:
    """
    Convert *text* into a filesystem-safe slug.
    Keeps alphanumeric characters, underscores, and hyphens; replaces anything
    else with underscores; collapses consecutive underscores; truncates to
    *max_len* characters.
    """
    slug = re.sub(r"[^\w\-]", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len] if slug else "session"


# ──────────────────────────────────────────────────────────────────────────────
# SessionState dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    Represents everything needed to save and restore a LabOpt session.

    pending_batch
        List of trials that have been *asked* from Optuna but whose lab
        results have not yet been *told* back.  Each entry is:
            {"trial_number": int, "params": {col: val, ...}}

    pending_planned_context
        The context values the user entered in the "Current Conditions"
        panel *before* asking for the pending batch.  These are used to
        pre-fill the context correction cells in the Batch Results Dialog
        when resuming a session.  ``None`` if no context variables exist
        or the user did not enter any planned context.

    project_name, owner, description
        Human-readable project identity fields.  ``project_name`` is the
        display name shown in the title bar and session browser.
        ``owner`` is the researcher / username; used to isolate sessions
        on disk when multiple users share a deployment.
        Old sessions without these fields get empty-string defaults.

    doe_state
        Optional DoE phase state.  None if the user has not run a DoE.
    """
    session_path: str          # absolute path to this .json file
    storage_path: str          # absolute path to the Optuna SQLite .db file
    study_name: str            # unique Optuna study name
    csv_path: str              # path to the experiment data CSV
    study_config: StudyConfig
    pending_batch: Optional[List[dict]] = None
    pending_planned_context: Optional[dict] = None   # {col_name: float, ...}

    # ── Human-readable project identity ────────────────────────────────────
    project_name: str = ""     # display name, e.g. "Perovskite LHS Study"
    owner: str = ""            # researcher / username
    description: str = ""     # optional longer description

    # ── DoE phase ──────────────────────────────────────────────────────────
    doe_state: Optional[DoEState] = None

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_path": self.session_path,
            "storage_path": self.storage_path,
            "study_name": self.study_name,
            "csv_path": self.csv_path,
            "study_config": self.study_config.to_dict(),
            "pending_batch": self.pending_batch,
            "pending_planned_context": self.pending_planned_context,
            "project_name": self.project_name,
            "owner": self.owner,
            "description": self.description,
            "doe_state": self.doe_state.to_dict() if self.doe_state is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        doe_raw = d.get("doe_state")
        return cls(
            session_path=d["session_path"],
            storage_path=d["storage_path"],
            study_name=d["study_name"],
            csv_path=d.get("csv_path", ""),
            study_config=StudyConfig.from_dict(d["study_config"]),
            pending_batch=d.get("pending_batch"),
            pending_planned_context=d.get("pending_planned_context"),
            project_name=d.get("project_name", ""),
            owner=d.get("owner", ""),
            description=d.get("description", ""),
            doe_state=DoEState.from_dict(doe_raw) if doe_raw is not None else None,
        )

    # ── Convenience ────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        """Human-readable name for title bar / menus."""
        if self.project_name:
            return self.project_name
        # Fallback for old sessions: strip prefix and suffix for readability
        return self.study_name


# ──────────────────────────────────────────────────────────────────────────────
# SessionManager
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Static helper class.  All methods are @staticmethod so the class can be
    used without instantiation (matching usage in main_window.py).
    """

    @staticmethod
    def create_new_session(
        config: StudyConfig,
        csv_path: str,
        session_dir: str,
        project_name: str = "",
        owner: str = "",
        description: str = "",
    ) -> SessionState:
        """
        Create a brand-new session directory structure and return the initial
        SessionState.  The SQLite DB is *not* created here — Optuna creates it
        lazily when build_study() is first called.

        Directory layout
        ----------------
        With owner:
            {session_dir}/{owner_slug}/{project_slug}_{timestamp}/
        Without owner (single-user fallback):
            {session_dir}/{project_slug}_{timestamp}/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build per-owner subdirectory if owner is given
        if owner:
            owner_slug = _slugify(owner, max_len=40)
            base = os.path.join(session_dir, owner_slug)
        else:
            base = session_dir

        if project_name:
            project_slug = _slugify(project_name, max_len=40)
            session_subdir = os.path.join(base, f"{project_slug}_{timestamp}")
        else:
            session_subdir = base  # flat layout, no subfolder per session

        os.makedirs(session_subdir, exist_ok=True)

        study_name = f"labopt_study_{timestamp}"
        storage_path = os.path.join(session_subdir, f"{study_name}.db")
        session_path = os.path.join(session_subdir, f"{study_name}_session.json")

        state = SessionState(
            session_path=session_path,
            storage_path=storage_path,
            study_name=study_name,
            csv_path=csv_path,
            study_config=config,
            pending_batch=None,
            project_name=project_name,
            owner=owner,
            description=description,
        )
        SessionManager.save(state)
        SessionManager._register_recent(state)
        return state

    @staticmethod
    def save(state: SessionState) -> None:
        """
        Write *state* to disk atomically (write to .tmp then rename) to avoid
        corruption if the process is killed mid-write.
        Also updates the registry entry's n_trials count.
        """
        tmp = state.session_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2)
        shutil.move(tmp, state.session_path)
        # Silently refresh the registry n_trials (best-effort, no crash on failure)
        try:
            SessionManager._refresh_registry_entry(state)
        except Exception:
            pass

    @staticmethod
    def load(session_path: str) -> SessionState:
        """
        Load a SessionState from a *_session.json file.

        Raises
        ------
        FileNotFoundError  if the SQLite DB referenced in the file is missing.
        """
        with open(session_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        state = SessionState.from_dict(data)
        if not os.path.exists(state.storage_path):
            raise FileNotFoundError(
                f"Optuna database not found: {state.storage_path}\n"
                "The session file may have been moved without its .db companion."
            )
        return state

    @staticmethod
    def list_recent_sessions(n: int = 5) -> List[dict]:
        """
        Return the *n* most recently used sessions from the global registry,
        filtered to only those whose session file still exists on disk.
        Each entry now includes project_name, owner, and n_trials.
        """
        registry_path = _get_registry_path()
        if not os.path.exists(registry_path):
            return []
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sessions: List[dict] = data.get("sessions", [])
            valid = [s for s in sessions if os.path.exists(s.get("session_path", ""))]
            return list(reversed(valid[-n:]))          # newest first
        except Exception:
            return []

    @staticmethod
    def list_all_sessions(base_dir: Optional[str] = None) -> List[dict]:
        """
        Recursively scan *base_dir* (default: get_default_session_dir()) for
        ``*_session.json`` files and return their metadata, sorted by timestamp
        descending.  Used by the Session Browser dialog.

        Returns
        -------
        List of dicts with keys:
            session_path, study_name, project_name, owner, description,
            timestamp, n_trials, doe_complete, has_pending_batch
        """
        root = base_dir or get_default_session_dir()
        results: List[dict] = []
        if not os.path.isdir(root):
            return results

        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith("_session.json"):
                    full_path = os.path.join(dirpath, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8") as fh:
                            d = json.load(fh)
                        # Count trials from DB (best-effort)
                        n_trials = SessionManager._count_trials_from_db(
                            d.get("storage_path", ""),
                            d.get("study_name", ""),
                        )
                        doe_state_raw = d.get("doe_state")
                        results.append({
                            "session_path": full_path,
                            "study_name": d.get("study_name", ""),
                            "project_name": d.get("project_name", ""),
                            "owner": d.get("owner", ""),
                            "description": d.get("description", ""),
                            "timestamp": d.get("study_name", "")[-15:],  # YYYYMMDD_HHMMSS
                            "n_trials": n_trials,
                            "doe_complete": (
                                doe_state_raw.get("complete", False)
                                if doe_state_raw else False
                            ),
                            "has_pending_batch": d.get("pending_batch") is not None,
                        })
                    except Exception:
                        pass  # skip unreadable files

        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    @staticmethod
    def mark_batch_pending(state: SessionState, param_dicts: List[dict]) -> None:
        """
        Record *param_dicts* as the currently pending batch, export their
        suggested parameters to *pending_batch.csv* for lab use, and save
        the session.

        Parameters
        ----------
        param_dicts
            List of ``{"trial_number": int, "params": dict}`` where *params*
            are the **constraint-enforced** values that will actually be shown
            to the user and run in the lab (not the raw Optuna internal params).
        """
        state.pending_batch = param_dicts
        SessionManager.save(state)

        # Write lab-friendly CSV
        session_dir = os.path.dirname(state.session_path)
        pending_csv = os.path.join(session_dir, "pending_batch.csv")
        rows = [{"trial_number": p["trial_number"], **p["params"]} for p in param_dicts]
        pd.DataFrame(rows).to_csv(pending_csv, index=False)

    @staticmethod
    def clear_pending_batch(state: SessionState) -> None:
        """Mark the pending batch as resolved and clean up the lab CSV."""
        state.pending_batch = None
        SessionManager.save(state)

        session_dir = os.path.dirname(state.session_path)
        pending_csv = os.path.join(session_dir, "pending_batch.csv")
        if os.path.exists(pending_csv):
            os.remove(pending_csv)

    @staticmethod
    def match_pending_to_csv(
        state: SessionState,
        df: pd.DataFrame,
        float_tol: float = 1e-6,
    ) -> Optional[Dict[int, List[float]]]:
        """
        Try to find rows in *df* that match each pending trial's parameters.

        Returns
        -------
        dict  {trial_number: [objective_value, ...]}  for matched rows, or
        None  if no rows matched at all.

        Partial matches (only some trials found) are included; the caller
        displays unmatched trials as blank cells in the results dialog.
        """
        if not state.pending_batch:
            return None

        obj_cols = [o.column_name for o in state.study_config.objectives]
        matches: Dict[int, List[float]] = {}

        for pending in state.pending_batch:
            trial_num = pending["trial_number"]
            params = pending["params"]

            for _, row in df.iterrows():
                # Row must have all objective values
                if any(pd.isna(row.get(c)) for c in obj_cols):
                    continue

                matched = True
                for col, expected in params.items():
                    if col not in row.index:
                        matched = False
                        break
                    actual = row[col]
                    # Always try numeric comparison first — pandas may upcast int
                    # columns to float64 when the DataFrame has mixed dtypes, so
                    # str("2000.0") != str("2000") even though the values match.
                    try:
                        if abs(float(actual) - float(expected)) > float_tol:
                            matched = False
                            break
                    except (TypeError, ValueError):
                        if str(actual).strip() != str(expected).strip():
                            matched = False
                            break

                if matched:
                    matches[trial_num] = [float(row[c]) for c in obj_cols]
                    break   # first matching row wins

        return matches if matches else None

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _register_recent(state: SessionState) -> None:
        """
        Add or update this session in the global registry.
        Registry entries now include project_name, owner, and n_trials.
        """
        registry_path = _get_registry_path()
        # Ensure the registry directory exists (important for Docker volume mounts)
        registry_dir = os.path.dirname(registry_path)
        if registry_dir:
            os.makedirs(registry_dir, exist_ok=True)

        sessions: List[dict] = []
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as fh:
                    sessions = json.load(fh).get("sessions", [])
            except Exception:
                sessions = []

        # Remove duplicate (same path)
        sessions = [s for s in sessions if s.get("session_path") != state.session_path]
        sessions.append({
            "session_path": state.session_path,
            "study_name": state.study_name,
            "project_name": state.project_name,
            "owner": state.owner,
            "timestamp": datetime.now().isoformat(),
            "n_trials": 0,
        })
        # Cap at 20 entries
        sessions = sessions[-20:]

        with open(registry_path, "w", encoding="utf-8") as fh:
            json.dump({"sessions": sessions}, fh, indent=2)

    @staticmethod
    def _refresh_registry_entry(state: SessionState) -> None:
        """
        Update n_trials in the registry for this session.  Best-effort only.
        """
        registry_path = _get_registry_path()
        if not os.path.exists(registry_path):
            return
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sessions: List[dict] = data.get("sessions", [])
            n = SessionManager._count_trials_from_db(
                state.storage_path, state.study_name
            )
            for entry in sessions:
                if entry.get("session_path") == state.session_path:
                    entry["n_trials"] = n
                    entry["project_name"] = state.project_name
                    entry["owner"] = state.owner
                    break
            with open(registry_path, "w", encoding="utf-8") as fh:
                json.dump({"sessions": sessions}, fh, indent=2)
        except Exception:
            pass

    @staticmethod
    def _count_trials_from_db(storage_path: str, study_name: str) -> int:
        """
        Return the number of completed Optuna trials in *storage_path*.
        Returns 0 on any error (DB missing, locked, etc.).
        """
        if not storage_path or not os.path.exists(storage_path):
            return 0
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            storage = f"sqlite:///{storage_path}"
            study = optuna.load_study(study_name=study_name, storage=storage)
            from optuna.trial import TrialState
            return len([t for t in study.trials if t.state == TrialState.COMPLETE])
        except Exception:
            return 0
