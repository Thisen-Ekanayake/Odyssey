"""
Package marker only. tools/ is a loose collection of independent scripts run
directly (``./airsim_venv/bin/python tools/<script>.py``), not a cohesive
API -- this file exists so other directories can reach ``tools.window_recorder``
via the same ``sys.path.insert(repo_root); from tools import ...`` pattern
already used for ``slam``.
"""
