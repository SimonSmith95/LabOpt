# -*- mode: python ; coding: utf-8 -*-
#
# LabOpt — PyInstaller build spec (one-dir mode)
#
# Build command:
#   pyinstaller LabOpt.spec --clean
#
# Output:
#   dist/LabOpt/LabOpt.exe   ← double-click to launch (no Python needed)
#   dist/LabOpt/             ← entire folder must be kept together
#
# To create a distributable archive:
#   Compress dist\LabOpt\ into a .zip and share that folder.
#
# Sessions are stored in the user's home directory (~/.labopt/ or
# %USERPROFILE%\.labopt\) so they persist across app updates.
#
# Key fix: Optuna uses Alembic for SQLite DB migrations. The migration
# scripts live in optuna/storages/_rdb/alembic/ and are data files,
# not Python modules — PyInstaller won't auto-detect them.
# collect_data_files('optuna') bundles the entire optuna data directory
# including those migration scripts.

from PyInstaller.utils.hooks import collect_data_files

# Collect optuna data files including .py migration scripts.
# include_py_files=True is required because Alembic's version scripts
# (optuna/storages/_rdb/alembic/versions/*.py) are loaded by Alembic via
# filesystem scanning — not by Python import — so they must be present as
# raw .py files at runtime, not only as compiled .pyc in the archive.
_optuna_datas = collect_data_files('optuna', include_py_files=True)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # ── Static assets bundled alongside the exe ───────────────────────
        ('DOCUMENTATION.md', '.'),   # in-app help viewer
        ('LabOpt_logo.png',  '.'),   # logo shown in the toolbar
    ] + _optuna_datas,               # ← Optuna data files (alembic migrations etc.)
    hiddenimports=[
        # ── Optuna + storage backend ──────────────────────────────────────
        'optuna',
        'optuna.samplers',
        'optuna.samplers._tpe',
        'optuna.samplers._nsgaii',
        'optuna.samplers._random',
        'optuna.samplers._gp',
        'optuna.storages',
        'optuna.storages._rdb',
        'optuna.storages._rdb.storage',
        'alembic',
        'alembic.config',
        'alembic.runtime.migration',
        'alembic.operations',
        'alembic.operations.ops',
        'alembic.script',
        'alembic.script.base',
        'alembic.ddl',
        'greenlet',
        'sqlalchemy',
        'sqlalchemy.dialects.sqlite',
        'sqlalchemy.dialects.sqlite.pysqlite',

        # ── Scientific / ML stack ─────────────────────────────────────────
        'sklearn',
        'sklearn.ensemble',
        'sklearn.inspection',
        'scipy',
        'scipy.stats',
        'scipy.spatial.distance',

        # ── Plotting ──────────────────────────────────────────────────────
        'matplotlib.backends.backend_qtagg',

        # ── Excel / file formats ──────────────────────────────────────────
        'openpyxl',

        # ── DoE strategies (soft dependency) ─────────────────────────────
        'pyDOE2',

        # ── System metrics + logging ──────────────────────────────────────
        'psutil',            # RSS / CPU monitoring in app_logger
        'logging.handlers',  # RotatingFileHandler used by app_logger
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages to reduce bundle size
        'tkinter',
        'PyQt5',
        'PyQt6',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # one-dir mode: binaries go into COLLECT
    name='LabOpt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no console window — GUI-only app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['LabOpt_logo.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LabOpt',
)
