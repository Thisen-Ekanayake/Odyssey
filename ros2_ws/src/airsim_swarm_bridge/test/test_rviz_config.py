#!/usr/bin/env python3
"""Every display class named in rviz/*.rviz must actually load on this machine.

    ./scripts/ros_enter.sh python3 ros2_ws/src/airsim_swarm_bridge/test/test_rviz_config.py

This exists because of a failure that cost a whole demo. `swarm_live.rviz` asked
for `octomap_rviz_plugins/OccupancyGrid` -- the *only* display of the cooperative
map -- and on this machine RViz cannot dlopen it:

    liboctomap_rviz_plugins.so: undefined symbol: _ZTIN7octomap13OcTreeStampedE

RViz logs that as an ERROR and carries on with the display silently missing, so
the demo opened, looked plausible, and showed no map. Nothing caught it:
`test_cooperative_mapping.py` runs with `rviz:=false`, and the assertion that a
config file *parses* would have passed too -- the YAML was always valid.

So this checks the two things that actually fail in practice:

  1. the config is well-formed YAML with the structure RViz expects, and
  2. every `Class:` in it resolves to a pluginlib-declared class whose shared
     library this machine can really dlopen.

(2) is done by walking the ament index rather than starting RViz, so it needs no
display and takes milliseconds. `ctypes.CDLL` reproduces exactly the dlopen RViz
does, undefined symbols and all.
"""
from __future__ import annotations

import ctypes
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

RVIZ_DIR = Path(__file__).resolve().parent.parent / "rviz"

# RViz's own pluginlib base classes register under these ament index resources.
RESOURCE_NAMES = (
    "rviz_common__pluginlib__plugin",
    "rviz_common__pluginlib__plugin__rviz_common__Display",
)

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    global _passed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed.append(name)
        print(f"  FAIL {name}  {detail}")
    return bool(cond)


def _prefixes() -> list[Path]:
    raw = os.environ.get("AMENT_PREFIX_PATH", "")
    return [Path(p) for p in raw.split(os.pathsep) if p]


def _plugin_libraries() -> dict[str, Path]:
    """Map 'package/ClassName' -> the .so pluginlib would dlopen for it."""
    out: dict[str, Path] = {}
    for prefix in _prefixes():
        index = prefix / "share" / "ament_index" / "resource_index"
        xmls: set[Path] = set()
        for resource in RESOURCE_NAMES:
            d = index / resource
            if not d.is_dir():
                continue
            for marker in d.iterdir():
                # The marker file holds newline-separated paths to plugin XMLs,
                # relative to the PREFIX (e.g. "share/<pkg>/plugins_description.xml"),
                # not to the package's own share dir.
                for line in marker.read_text().split("\n"):
                    line = line.strip()
                    if line:
                        xmls.add(Path(line) if line.startswith("/") else prefix / line)
        for xml in xmls:
            if not xml.is_file():
                continue
            try:
                root = ET.parse(xml).getroot()
            except ET.ParseError:
                continue
            libs = [root] if root.tag == "library" else root.findall(".//library")
            for lib in libs:
                path = lib.get("path", "")
                if not path:
                    continue
                stem = Path(path).name
                so = prefix / "lib" / (stem if stem.startswith("lib") else f"lib{stem}.so")
                if not so.exists():
                    so = prefix / "lib" / f"{stem}.so"
                for cls in lib.findall("class"):
                    name = cls.get("name") or cls.get("type", "")
                    if name:
                        out[name] = so
    return out


def main() -> int:
    configs = sorted(RVIZ_DIR.glob("*.rviz"))
    if not check("found rviz configs", bool(configs), f"none under {RVIZ_DIR}"):
        return 1

    libs = _plugin_libraries()
    # A machine with no ROS sourced would report every class as missing, which is
    # a confusing way to say "your environment is wrong". Say that instead.
    if not check("ament index lists RViz plugins", bool(libs),
                 "no plugin XMLs found -- is the ROS environment sourced? "
                 f"AMENT_PREFIX_PATH={os.environ.get('AMENT_PREFIX_PATH', '')!r}"):
        return 1

    loaded: dict[Path, str | None] = {}
    for config in configs:
        try:
            doc = yaml.safe_load(config.read_text())
        except yaml.YAMLError as exc:
            check(f"{config.name} parses as YAML", False, str(exc))
            continue
        check(f"{config.name} parses as YAML", True)

        displays = (doc.get("Visualization Manager") or {}).get("Displays")
        if not check(f"{config.name} has a Displays list", isinstance(displays, list),
                     f"got {type(displays).__name__}"):
            continue

        classes = sorted({d["Class"] for d in displays if isinstance(d, dict) and d.get("Class")})
        check(f"{config.name} declares display classes", bool(classes))

        for cls in classes:
            so = libs.get(cls)
            if not check(f"{config.name}: {cls} is a known plugin class", so is not None,
                         "not declared by any package in the ament index"):
                continue
            if so not in loaded:
                try:
                    ctypes.CDLL(str(so))
                    loaded[so] = None
                except OSError as exc:
                    loaded[so] = str(exc)
            check(f"{config.name}: {cls} library loads", loaded[so] is None,
                  f"{so}: {loaded[so]}")

    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        print("failed: " + ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
