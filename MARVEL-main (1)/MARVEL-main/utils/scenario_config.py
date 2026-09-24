"""Scenario configuration loading and validation for the extensible simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ScenarioConfigError(ValueError):
    """Raised when a scenario configuration is invalid."""


def _ranges_overlap(left: list[int], right: list[int]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])


def load_and_validate_scenario(scenario_file: str | Path) -> Dict[str, Any]:
    """Load a YAML scenario and resolve module config paths relative to it."""
    path = Path(scenario_file).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    for field in ("scenario", "environment", "robots", "tasks"):
        if field not in config:
            raise ScenarioConfigError(f"Missing required field: {field}")
    if not isinstance(config["robots"], list) or not config["robots"]:
        raise ScenarioConfigError("robots must be a non-empty list")
    if not isinstance(config["tasks"], list):
        raise ScenarioConfigError("tasks must be a list")

    ranges: list[list[int]] = []
    for robot_group in config["robots"]:
        robot_range = robot_group.get("id_range")
        if not isinstance(robot_range, list) or len(robot_range) != 2:
            raise ScenarioConfigError("each robot group needs id_range: [start, end]")
        if robot_range[0] > robot_range[1] or robot_range[0] < 0:
            raise ScenarioConfigError(f"invalid robot id range: {robot_range}")
        if any(_ranges_overlap(robot_range, other) for other in ranges):
            raise ScenarioConfigError(f"overlapping robot id range: {robot_range}")
        ranges.append(robot_range)

    for module in ("dynamics", "sensor", "communication"):
        section = config.get(module)
        if not section:
            continue
        config_file = Path(section.get("config_file", ""))
        if not config_file.is_absolute():
            config_file = (path.parent / config_file).resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"{module} config not found: {config_file}")
        section["config_file"] = str(config_file)
        with config_file.open("r", encoding="utf-8") as handle:
            section["params"] = yaml.safe_load(handle) or {}

    # Resolve optional map_file relative to the scenario YAML location
    env = config["environment"]
    map_file = env.get("map_file")
    if map_file:
        map_path = Path(map_file)
        if not map_path.is_absolute():
            map_path = (path.parent / map_file).resolve()
        if not map_path.exists():
            raise FileNotFoundError(f"map_file not found: {map_path}")
        env["map_file"] = str(map_path)

    # Resolve optional map_dir (MARVEL-native geometry mode) the same way.
    map_dir = env.get("map_dir")
    if map_dir:
        map_root = Path(map_dir)
        if not map_root.is_absolute():
            map_root = (path.parent / map_dir).resolve()
        if not map_root.is_dir():
            raise FileNotFoundError(f"map_dir not found: {map_root}")
        env["map_dir"] = str(map_root)

    # geometry_mode is explicit; never inferred from the presence of a map.
    mode = str(env.get("geometry_mode", "extended")).strip().lower()
    if mode not in ("extended", "marvel_native"):
        raise ScenarioConfigError(
            "environment.geometry_mode must be 'extended' or "
            f"'marvel_native', got {mode!r}"
        )
    env["geometry_mode"] = mode

    if mode == "marvel_native" and not env.get("map_dir"):
        raise ScenarioConfigError(
            "geometry_mode 'marvel_native' requires environment.map_dir"
        )

    config["_source_file"] = str(path)
    return config


def robot_count(config: Dict[str, Any]) -> int:
    """Return the number of robots described by a validated scenario."""
    return sum(group["id_range"][1] - group["id_range"][0] + 1 for group in config["robots"])
