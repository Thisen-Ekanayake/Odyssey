"""
Reusable "Desmos-style" 3D-view rotation control for an Open3D SceneWidget:
a small panel (manual azimuth slider, "Auto-rotate 360deg" checkbox, speed
number entry) that orbits the widget's camera around a fixed elevation/radius,
centered on a given bounds' center -- dragging the slider or the animation
both just change the azimuth angle fed to the same camera.look_at() call.

Originally lived only in swarm_converge_viz.py; pulled out here so every
LiDAR/SLAM map viewer gets it, not just that one script. Deliberately only
wired into scenes actually showing sensor-derived data (a LiDAR point-cloud
map, a SLAM map) -- chase-cam feeds are plain image widgets with no camera
to orbit, and formation-geometry-only scenes (no sensor data, e.g. the
edge-to-center segmentation lines) don't get it either.

Usage (after the SceneWidget's own setup_camera() call):

    self.rotation = scene_rotation.RotationPanel(
        self.win, self.widget, bounds, is_running=lambda: self._running)

and from that window's own set_on_layout callback, once the SceneWidget's
own frame is set:

    self.rotation.layout(self.widget.frame)
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import open3d.visualization.gui as gui  # type: ignore

PANEL_SIZE = (230, 160)
DEFAULT_ELEVATION_DEG = 35.0   # fixed pitch of the orbit; azimuth is what rotates
DEFAULT_ROTATE_SPEED = 2.0    # deg/s for the auto-rotate animation
ROTATE_HZ = 20                 # animation tick rate


class RotationPanel:
    def __init__(self, window, widget, bounds, *, is_running,
                 center=None,
                 elevation_deg: float = DEFAULT_ELEVATION_DEG,
                 rotate_speed: float = DEFAULT_ROTATE_SPEED,
                 panel_size=PANEL_SIZE):
        self._window = window
        self._widget = widget
        self._is_running = is_running
        self._panel_size = panel_size

        # Radius sized off the XY footprint (much larger than the Z range)
        # so the whole scene stays framed at any azimuth. `bounds` is often
        # padded well past where the actual geometry sits (e.g. enough Z
        # range to cover a drone's full climb, not just its ground-scan
        # returns), so its own center can be a poor look-at target -- pass
        # `center` explicitly to orbit around the real data instead; this is
        # what keeps the scene framed dead-center on screen, including while
        # auto-rotating (no way to nudge it back once the animation's own
        # look_at() call overwrites any manual pan on the next tick).
        extent = bounds.get_extent()
        self._center = tuple(center) if center is not None else tuple(bounds.get_center())
        self._orbit_radius = 0.9 * math.hypot(extent[0], extent[1])
        self._elevation_deg = elevation_deg
        self._azimuth_deg = 0.0
        self._auto_rotate = False
        self._rotate_speed = rotate_speed

        self._build_panel()
        self.apply_azimuth(self._azimuth_deg)   # own consistent initial view, not setup_camera's default

        threading.Thread(target=self._rotate_loop, daemon=True).start()

    # ---- panel + layout -----------------------------------------------

    def _build_panel(self):
        em = self._window.theme.font_size
        panel = gui.Vert(0.4 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        panel.background_color = gui.Color(0.1, 0.1, 0.1, 0.75)

        panel.add_child(gui.Label("View Rotation"))

        self.azimuth_slider = gui.Slider(gui.Slider.DOUBLE)
        self.azimuth_slider.set_limits(0.0, 360.0)
        self.azimuth_slider.double_value = self._azimuth_deg
        self.azimuth_slider.set_on_value_changed(self._on_azimuth_changed)
        panel.add_child(gui.Label("Rotate (drag)"))
        panel.add_child(self.azimuth_slider)

        self.auto_rotate_checkbox = gui.Checkbox("Auto-rotate 360°")
        self.auto_rotate_checkbox.set_on_checked(self._on_auto_rotate_toggled)
        panel.add_child(self.auto_rotate_checkbox)

        self.speed_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.speed_edit.set_limits(1.0, 90.0)
        self.speed_edit.double_value = self._rotate_speed
        self.speed_edit.set_on_value_changed(self._on_speed_changed)
        panel.add_child(gui.Label("Speed (deg/s)"))
        panel.add_child(self.speed_edit)

        self.panel = panel
        self._window.add_child(panel)

    def layout(self, content_rect, margin: int = 10) -> None:
        """Position the panel in `content_rect`'s top-right corner. Call
        from the owning window's set_on_layout, after the SceneWidget's own
        frame is set -- content_rect is typically that widget's own .frame
        (e.g. a right-hand split), not necessarily the whole window's."""
        pw, ph = self._panel_size
        self.panel.frame = gui.Rect(
            content_rect.x + content_rect.width - pw - margin,
            content_rect.y + margin, pw, ph)

    # ---- camera math ----------------------------------------------------

    def _camera_vectors(self, azimuth_deg: float):
        """(center, eye, up) for an orbit camera at the given azimuth, fixed
        elevation/radius -- eye moves on a circle around the scene center."""
        az = math.radians(azimuth_deg)
        el = math.radians(self._elevation_deg)
        horiz = self._orbit_radius * math.cos(el)
        height = self._orbit_radius * math.sin(el)
        cx, cy, cz = self._center
        center = np.array([cx, cy, cz], dtype=np.float32)
        eye = np.array([
            cx + horiz * math.cos(az),
            cy + horiz * math.sin(az),
            cz - height,   # NED: -Z is up
        ], dtype=np.float32)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return center, eye, up

    def apply_azimuth(self, azimuth_deg: float) -> None:
        center, eye, up = self._camera_vectors(azimuth_deg)
        self._widget.scene.camera.look_at(center, eye, up)

    # ---- widget callbacks -------------------------------------------------

    def _on_azimuth_changed(self, value):
        self._azimuth_deg = value
        self.apply_azimuth(value)

    def _on_auto_rotate_toggled(self, checked):
        self._auto_rotate = checked

    def _on_speed_changed(self, value):
        self._rotate_speed = value

    # ---- background animation ----------------------------------------

    def _rotate_loop(self):
        """Background tick for the auto-rotate animation; a no-op spin while
        auto-rotate is off. GUI mutation is marshaled to the main thread."""
        dt = 1.0 / ROTATE_HZ
        while self._is_running():
            if self._auto_rotate:
                self._azimuth_deg = (self._azimuth_deg + self._rotate_speed * dt) % 360.0
                az = self._azimuth_deg

                def _update(az=az):
                    self.apply_azimuth(az)
                    self.azimuth_slider.double_value = az

                gui.Application.instance.post_to_main_thread(self._window, _update)
            time.sleep(dt)
