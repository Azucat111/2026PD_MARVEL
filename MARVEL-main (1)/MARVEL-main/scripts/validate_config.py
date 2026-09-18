"""Validate a scenario file and print a compact summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.scenario_config import load_and_validate_scenario, robot_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_file")
    args = parser.parse_args()
    try:
        config = load_and_validate_scenario(args.scenario_file)
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"VALID: {Path(args.scenario_file).name}")
    print(f"robots={robot_count(config)} tasks={len(config['tasks'])} "
          f"dynamic_obstacles={len(config.get('dynamic_obstacles', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
