"""Lightweight, replaceable 2-D vehicle dynamics models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml


def _angle_delta(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


class DynamicsModel:
    """Base interface for a dynamics model."""

    def step(self, current_position, final_position, theta_current, theta_desired,
             v_current=0.0, dt=0.1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError


class KinematicSimple(DynamicsModel):
    """MARVEL-compatible constant-speed planar model."""

    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.velocity = float(params.get("velocity", 1.0))
        self.yaw_rate = float(params.get("yaw_rate", 35.0))

    def step(self, current_position, final_position, theta_current, theta_desired,
             v_current=0.0, dt=0.1):
        current = np.asarray(current_position, dtype=float)
        target = np.asarray(final_position, dtype=float)
        distance = float(np.linalg.norm(target - current))
        travel_dt = distance / max(self.velocity, 1e-6)
        max_delta = self.yaw_rate * travel_dt
        delta = _angle_delta(float(theta_desired), float(theta_current))
        applied = np.clip(delta, -max_delta, max_delta)
        heading = (float(theta_current) + applied) % 360.0
        if distance > 1e-9:
            direction = (target - current) / distance
            position = current + direction * min(distance, self.velocity * dt)
        else:
            position = current.copy()
        return (
            {"position": position, "velocity": self.velocity, "heading": heading,
             "angular_velocity": applied / max(dt, 1e-6)},
            {"speed_saturated": False, "turn_saturated": abs(applied - delta) > 1e-6,
             "min_radius_violated": False},
        )


class KinematicWithAccel(DynamicsModel):
    """Acceleration and turn-rate constrained planar model."""

    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.v_min = float(params.get("v_min", 0.3))
        self.v_max = float(params.get("v_max", 2.5))
        self.a_max = float(params.get("a_max", 1.2))
        self.omega_max = float(params.get("omega_max", 45.0))
        self.alpha_max = float(params.get("alpha_max", 30.0))
        self.min_turn_radius = float(params.get("min_turn_radius", 0.8))

    def step(self, current_position, final_position, theta_current, theta_desired,
             v_current=0.0, dt=0.1, v_desired=None):
        current = np.asarray(current_position, dtype=float)
        target = np.asarray(final_position, dtype=float)
        desired_speed = self.v_max if v_desired is None else float(v_desired)
        speed = np.clip(desired_speed, float(v_current) - self.a_max * dt,
                        float(v_current) + self.a_max * dt)
        speed_clipped = float(np.clip(speed, self.v_min, self.v_max))
        desired_omega = _angle_delta(float(theta_desired), float(theta_current)) / max(dt, 1e-6)
        omega = float(np.clip(desired_omega, -self.omega_max, self.omega_max))
        turn_saturated = abs(omega - desired_omega) > 1e-6
        radius_violated = abs(omega) > 1e-6 and speed_clipped / abs(math.radians(omega)) < self.min_turn_radius
        if radius_violated:
            speed_clipped = min(speed_clipped, self.min_turn_radius * abs(math.radians(omega)))
        heading = (float(theta_current) + omega * dt) % 360.0
        position = current + speed_clipped * dt * np.array(
            [math.cos(math.radians(heading)), math.sin(math.radians(heading))])
        return (
            {"position": position, "velocity": speed_clipped, "heading": heading,
             "angular_velocity": omega},
            {"speed_saturated": abs(speed - speed_clipped) > 1e-6,
             "turn_saturated": turn_saturated,
             "min_radius_violated": radius_violated},
        )


def create_dynamics_model(config_file: str | Path | Dict[str, Any]) -> DynamicsModel:
    params = config_file if isinstance(config_file, dict) else _read_yaml(config_file)
    model_type = params.get("model_type", "kinematic_2d")
    model_params = params.get("params", params)
    if model_type in ("kinematic_accel", "kinematic_with_accel"):
        return KinematicWithAccel(model_params)
    return KinematicSimple(model_params)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
