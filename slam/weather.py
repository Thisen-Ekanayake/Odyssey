"""
Weather control for the benchmark grid.

Two things worth stating plainly, because they shape the whole study:

1. AirSim's weather is a **rendering** effect. ``simSetWeatherParameter`` drives
   particle systems and post-process volumes in UE4, so it changes what the
   cameras see and nothing else.
2. The LiDAR is a raycast against collision geometry. Rain and fog particles
   have no collision, so **LiDAR returns are identical in every condition**.
   ``tools/probe_setup.py`` measures this rather than asserting it.

That asymmetry is why ``slam.degradation`` exists: camera degradation comes
free from the renderer, LiDAR degradation has to be modelled on top. Both are
driven from the same ``WeatherCondition`` so the severities stay matched.

This module holds the only ``airsim`` import in the weather path; ``config``
stores parameter names as strings so the offline benchmark can import the
condition table without the RPC stack.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import airsim

from . import config as cfg
from .config import WeatherCondition

# UE4 particle systems and fog volumes ramp in over roughly a second; capturing
# before they settle would put a transient into the first frames of a dataset.
SETTLE_SECONDS = 2.5


def _resolve(param_name: str) -> int:
    """Map a ``config`` string key onto an ``airsim.WeatherParameter`` value."""
    try:
        return getattr(airsim.WeatherParameter, param_name)
    except AttributeError as exc:
        valid = [k for k in vars(airsim.WeatherParameter) if not k.startswith("_")]
        raise ValueError(
            f"unknown weather parameter {param_name!r}; valid names: {sorted(valid)}"
        ) from exc


def reset(client: airsim.MultirotorClient) -> None:
    """Zero every weather parameter and disable the weather system.

    Zeroing before disabling matters: AirSim keeps the last value per parameter,
    so a later ``simEnableWeather(True)`` would silently restore the previous
    condition and contaminate the next dataset.
    """
    for name in ("Rain", "Roadwetness", "Snow", "RoadSnow",
                 "MapleLeaf", "RoadLeaf", "Dust", "Fog"):
        try:
            client.simSetWeatherParameter(_resolve(name), 0.0)
        except Exception:
            pass
    client.simEnableWeather(False)


def apply(client: airsim.MultirotorClient, condition: WeatherCondition,
          settle: bool = True) -> None:
    """Put the sim into ``condition``, starting from a known-clear state."""
    reset(client)

    if not condition.airsim_params:
        if settle:
            time.sleep(SETTLE_SECONDS)
        return

    client.simEnableWeather(True)
    for name, value in condition.airsim_params.items():
        client.simSetWeatherParameter(_resolve(name), float(value))

    if settle:
        time.sleep(SETTLE_SECONDS)


@contextmanager
def weather(client: airsim.MultirotorClient, condition: WeatherCondition):
    """Scoped weather, cleared on the way out even if the body raises."""
    apply(client, condition)
    try:
        yield condition
    finally:
        reset(client)


def get(name: str) -> WeatherCondition:
    """Look up a condition by name, with a helpful error for typos."""
    try:
        return cfg.WEATHER_CONDITIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown condition {name!r}; available: {cfg.BENCHMARK_ORDER}"
        ) from exc
