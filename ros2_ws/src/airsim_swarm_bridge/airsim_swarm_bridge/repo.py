"""Makes the repo's existing non-ROS Python importable from inside ROS nodes.

``swarm/`` and ``slam/`` predate this workspace and are not ament packages -- the
``swarm`` scripts in particular import each other by flat module name
(``from swarm_comms import ...``), which only works if that directory is on
``sys.path``. Rather than fork them into the ROS package, the nodes call
:func:`add_swarm_to_path` / :func:`add_repo_to_path` and import the originals.
There is then exactly one definition of the drone layout and the manoeuvres, and
a fix to either is a fix everywhere.

The ``airsim`` dependency is handled by :func:`airsim_api.install_as_airsim`,
which aliases the tornado-free shim into ``sys.modules`` before those imports run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["repo_root", "add_repo_to_path", "add_swarm_to_path",
           "import_swarm_module", "settings_json_path"]

_ENV = "AIRSIM_SWARM_REPO"
_DEFAULT = "/ml/airsim_swarm"


def repo_root() -> Path:
    """Locate the repo: ``$AIRSIM_SWARM_REPO``, else walk up from this file, else /ml.

    The walk-up matters because the ROS workspace lives *inside* the repo
    (``<repo>/ros2_ws/src/...``), so the answer is usually four levels up -- but
    colcon's ``--symlink-install`` means ``__file__`` can resolve through the
    install tree, hence the explicit marker check rather than a fixed count.
    """
    env = os.environ.get(_ENV)
    if env and (Path(env) / "CLAUDE.md").exists():
        return Path(env)

    for base in (Path(__file__).resolve(), Path(__file__).absolute()):
        for parent in base.parents:
            if (parent / "settings.json").exists() and (parent / "swarm").is_dir():
                return parent

    fallback = Path(_DEFAULT)
    if (fallback / "settings.json").exists():
        return fallback
    raise RuntimeError(
        f"cannot locate the airsim_swarm repo. Set {_ENV} (scripts/ros_env.sh does "
        f"this) or run from inside the checkout."
    )


def add_repo_to_path() -> Path:
    """Put the repo root on ``sys.path`` so ``import slam.config`` works."""
    root = repo_root()
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return root


def add_swarm_to_path() -> Path:
    """Put ``<repo>/swarm`` on ``sys.path`` for its flat-module imports."""
    d = repo_root() / "swarm"
    p = str(d)
    if p not in sys.path:
        sys.path.insert(0, p)
    return d


def import_swarm_module(name: str):
    """Import one ``swarm/`` module with the ``airsim`` shim already in place.

    ``name`` is the bare module name, e.g. ``"swarm_comms"``.
    """
    from .airsim_api import install_as_airsim

    install_as_airsim()
    add_swarm_to_path()
    return __import__(name)


def settings_json_path() -> Path:
    """The ``settings.json`` the simulator was launched with."""
    return repo_root() / "settings.json"
