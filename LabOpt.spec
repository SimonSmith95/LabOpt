# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('DOCUMENTATION.md', '.'), ('LabOpt_logo.png', '.')],
    hiddenimports=['optuna', 'optuna.samplers', 'optuna.storages', 'alembic', 'alembic.config', 'alembic.runtime.migration', 'alembic.operations', 'alembic.operations.ops', 'alembic.script', 'alembic.ddl', 'greenlet', 'sklearn', 'scipy', 'openpyxl', 'sqlalchemy.dialects.sqlite', 'matplotlib.backends.backend_qtagg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LabOpt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
