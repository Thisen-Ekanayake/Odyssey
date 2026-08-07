"""
Adverse-weather degradation model for the LiDAR.

**Why this file exists.** AirSim's rain and fog are rendering effects. The
LiDAR is a raycast against collision geometry, and rain/fog particles have no
collision, so LiDAR returns are bit-identical in every weather condition
(``tools/verify_dataset.py`` measures this). Without a model, the weather half
of the study would only ever say "the cameras got worse and the LiDAR did not",
which is a statement about AirSim, not about sensors.

So camera degradation comes from the renderer and LiDAR degradation is modelled
here, both driven from the same ``WeatherCondition`` so the severities match.

**This is the study's main modelling assumption.** It is applied as a
post-process on recorded scans, which means (a) no extra flights are needed,
(b) every dataset can be scored both raw and degraded, and (c) all the
coefficients live in one table below where they can be argued with.

Physical basis
--------------
Beer-Lambert attenuation over the round trip, inside the LiDAR range equation::

    P_received  ∝  rho * exp(-2 * alpha * r) / r^2

A point is detected when the returned power clears the receiver threshold.
Normalising by the clear-weather maximum range gives a detection probability
with no free gain constant::

    p_detect(r) = min(1, (rho/rho_0) * (R_max/r)^2 * exp(-2*alpha*r))

Extinction coefficient ``alpha`` [1/m]:

* **Fog** -- Koschmieder: ``alpha = 3.912 / V`` for meteorological visibility
  ``V``. Fog droplets are large compared with 905 nm, so Mie scattering is
  roughly wavelength-independent and the visible-band relation carries over.
* **Rain** -- power law in rain rate ``R`` [mm/h]: ``alpha = k_rain * R^0.67``.
  ``k_rain`` is calibrated so heavy rain (40 mm/h) costs ~35% of maximum range,
  matching the 20-40% reduction reported for 905 nm automotive LiDAR in the
  adverse-weather literature.

Rain turns out to hurt LiDAR far less than fog does, which is the physically
correct answer and one of the more interesting things the study can show.

Four effects are modelled:

1. **Attenuation dropout** -- the range-dependent detection probability above.
2. **Range noise** -- pulse broadening from scattering inflates the ranging
   error with both ``alpha`` and ``r``.
3. **Backscatter clutter** -- droplets and fog return spurious near-range
   points, which is what actually breaks scan matching: they are dense,
   coherent-looking, and sit right where the algorithm expects real structure.
4. **Rain speckle** -- range-independent dropout from individual droplets
   occluding the beam.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as cfg
from .config import WeatherCondition

__all__ = ["COEFFICIENTS", "DegradationModel", "for_condition"]


# --------------------------------------------------------------------------
# The assumption layer. Everything contestable about this model is here.
# --------------------------------------------------------------------------
COEFFICIENTS = {
    # Rain extinction: alpha[1/m] = K_RAIN * R[mm/h] ** RAIN_EXPONENT.
    # Calibrated so 40 mm/h costs ~35% of clear-weather max range.
    "K_RAIN": 4.0e-4,
    "RAIN_EXPONENT": 0.67,

    # Fog extinction from visibility: alpha = KOSCHMIEDER / V.
    "KOSCHMIEDER": 3.912,

    # Nominal target reflectivity the detection threshold is normalised to.
    # Points are given a lognormal spread about this, so marginal returns
    # flicker in and out rather than cutting off at a hard range.
    "RHO_NOMINAL": 0.20,
    "RHO_SIGMA": 0.45,

    # Ranging noise: sigma_r = SIGMA_BASE + SIGMA_SCATTER * alpha * r  [m]
    "SIGMA_BASE": 0.02,          # clear-weather sensor noise
    "SIGMA_SCATTER": 0.35,

    # Backscatter clutter. Count scales with alpha; points fall off with 1/e
    # distance CLUTTER_DECAY / alpha, so clutter concentrates near the sensor
    # exactly when extinction is strong.
    #
    # Calibrated so dense fog (alpha = 0.065 /m) contributes clutter worth
    # ~25% of the surviving scan, matching reports that a large minority of
    # returns in dense fog come from the medium rather than from surfaces.
    # This is the effect that actually breaks scan matching -- the dropout
    # merely thins the cloud, whereas clutter adds dense, plausible-looking
    # structure that moves with the sensor.
    "CLUTTER_GAIN": 1.2,         # points per (1/m of alpha) per original point
    "CLUTTER_DECAY": 0.5,
    "CLUTTER_MIN_RANGE": 0.8,    # m, inside the sensor's blind zone

    # Rain speckle: per-point dropout probability = SPECKLE_GAIN * R[mm/h].
    "SPECKLE_GAIN": 0.0022,
}


def alpha_rain(rain_rate_mmph: float) -> float:
    """Extinction coefficient [1/m] for a rain rate in mm/h."""
    if rain_rate_mmph <= 0:
        return 0.0
    return COEFFICIENTS["K_RAIN"] * rain_rate_mmph ** COEFFICIENTS["RAIN_EXPONENT"]


def alpha_fog(visibility_m: float) -> float:
    """Extinction coefficient [1/m] from meteorological visibility (Koschmieder)."""
    if not math.isfinite(visibility_m) or visibility_m <= 0:
        return 0.0
    return COEFFICIENTS["KOSCHMIEDER"] / visibility_m


@dataclass
class DegradationModel:
    """Applies weather degradation to a sensor-local LiDAR scan.

    Callable with ``(N,3)`` sensor-frame points, returning a degraded ``(N',3)``.
    That signature is exactly ``DatasetSource(lidar_transform=...)``, so the
    benchmark can switch degradation on and off without touching the SLAM code.
    """
    alpha: float = 0.0                 # total extinction coefficient [1/m]
    rain_rate_mmph: float = 0.0
    fog_visibility_m: float = math.inf
    max_range: float = cfg.LIDAR_RANGE
    seed: int | None = 0
    label: str = "clear"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # -- diagnostics -------------------------------------------------------

    def detection_probability(self, r: np.ndarray | float) -> np.ndarray:
        """p_detect at range r for a nominally-reflective target."""
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        safe = np.maximum(r, 1e-3)
        p = (self.max_range / safe) ** 2 * np.exp(-2.0 * self.alpha * safe)
        return np.clip(p, 0.0, 1.0)

    def effective_range(self, threshold: float = 0.5) -> float:
        """Range at which detection probability falls below ``threshold``.

        The headline number for "how much reach did this weather cost".
        """
        r = np.linspace(1.0, self.max_range, 2000)
        p = self.detection_probability(r)
        below = np.nonzero(p < threshold)[0]
        return float(r[below[0]]) if len(below) else float(self.max_range)

    # -- the model ---------------------------------------------------------

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.0 and self.rain_rate_mmph <= 0.0:
            return points
        if points is None or len(points) == 0:
            return points

        pts = np.asarray(points, dtype=np.float32)
        r = np.linalg.norm(pts, axis=1)
        valid = r > 1e-3
        pts, r = pts[valid], r[valid]
        if len(pts) == 0:
            return pts

        n = len(pts)

        # 1. attenuation dropout, with per-point reflectivity spread so the
        #    cutoff is a soft shoulder rather than a hard sphere
        rho = self._rng.lognormal(0.0, COEFFICIENTS["RHO_SIGMA"], n)
        p_det = np.clip(rho * self.detection_probability(r), 0.0, 1.0)
        keep = self._rng.random(n) < p_det

        # 2. rain speckle -- droplet occlusion, independent of range
        if self.rain_rate_mmph > 0:
            p_speckle = min(0.9, COEFFICIENTS["SPECKLE_GAIN"] * self.rain_rate_mmph)
            keep &= self._rng.random(n) >= p_speckle

        pts, r = pts[keep], r[keep]

        # 3. range noise -- scattering broadens the return pulse
        if len(pts):
            sigma = COEFFICIENTS["SIGMA_BASE"] + COEFFICIENTS["SIGMA_SCATTER"] * self.alpha * r
            scale = 1.0 + self._rng.normal(0.0, sigma) / np.maximum(r, 1e-3)
            pts = (pts * scale[:, None]).astype(np.float32)

        # 4. backscatter clutter -- spurious near-range returns off the
        #    scattering medium itself. These matter more than the dropout:
        #    they look like real structure to a scan matcher.
        clutter = self._clutter(n)
        if len(clutter):
            pts = np.vstack([pts, clutter]) if len(pts) else clutter

        return np.ascontiguousarray(pts, dtype=np.float32)

    def _clutter(self, n_original: int) -> np.ndarray:
        n_clutter = int(COEFFICIENTS["CLUTTER_GAIN"] * self.alpha * n_original)
        if n_clutter <= 0:
            return np.empty((0, 3), dtype=np.float32)

        # Exponentially distributed range: backscatter is dominated by the
        # medium closest to the sensor.
        decay = COEFFICIENTS["CLUTTER_DECAY"] / max(self.alpha, 1e-6)
        r = COEFFICIENTS["CLUTTER_MIN_RANGE"] + self._rng.exponential(decay, n_clutter)
        r = r[r < self.max_range]
        if len(r) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Directions spread over the sensor's actual field of view.
        az = self._rng.uniform(-np.pi, np.pi, len(r))
        el = np.radians(self._rng.uniform(cfg.LIDAR_VFOV_LOWER, cfg.LIDAR_VFOV_UPPER, len(r)))
        d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)], axis=1)
        return (d * r[:, None]).astype(np.float32)

    def describe(self) -> str:
        return (f"{self.label}: alpha={self.alpha:.5f} /m, "
                f"effective range {self.effective_range():.1f} m "
                f"({100 * self.effective_range() / self.max_range:.0f}% of clear)")


def for_condition(condition: WeatherCondition, seed: int | None = 0) -> DegradationModel:
    """Build the model matching a benchmark weather condition.

    Rain and fog extinction add: a condition specifying both would be attenuated
    by both, which is the physically right composition.
    """
    a = alpha_rain(condition.rain_rate_mmph) + alpha_fog(condition.fog_visibility_m)
    return DegradationModel(
        alpha=a,
        rain_rate_mmph=condition.rain_rate_mmph,
        fog_visibility_m=condition.fog_visibility_m,
        max_range=cfg.LIDAR_RANGE,
        seed=seed,
        label=condition.name,
    )
