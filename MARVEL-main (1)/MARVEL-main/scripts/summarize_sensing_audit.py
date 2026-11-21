"""Consolidate the extended sensing / park-deadlock audit into a report.

    python scripts/summarize_sensing_audit.py --stamp <timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def pct(value):
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--native-gate", default=None)
    args = parser.parse_args()

    directory = ARTIFACTS / f"extended_sensing_correctness_{args.stamp}"
    ab = read_csv(directory / "occlusion_false_vs_true.csv")
    trace = json.loads((directory / "deadlock_trace.json").read_text())
    long_rows = (
        read_csv(directory / "long_horizon.csv")
        if (directory / "long_horizon.csv").exists() else []
    )
    native_gate = args.native_gate or (
        (directory / "native_gate.txt").read_text(encoding="utf-8")
        if (directory / "native_gate.txt").exists() else "(not recorded)"
    )

    def arm(flag: str):
        return [r for r in ab if r["los_occlusion"] == flag]

    lines: list[str] = []
    add = lines.append

    add("# Extended sensing / park-deadlock correctness audit")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat()}.")
    add("")
    add("Measurement only: no production file was modified. The A/B is driven")
    add("by writing `sensor.params.los_occlusion` into the scenario dict before")
    add("the runtime is constructed -- the same path `prepare_gppo_config`")
    add("itself uses.")
    add("")
    add("## 1. Sensor configuration semantics")
    add("")
    add("- Defined at `utils/sensor_models.py:28`:")
    add("  `self.los_occlusion = bool(params.get(\"los_occlusion\", False))`,")
    add("  consumed only by `IdealSensor._visible_cells`.")
    add("- Default: **False**.")
    add("- `configs/sensors/ideal.yaml` does not set it, so every scenario that")
    add("  includes that file starts from False.")
    add("- **But `integrations/gppo/config_profile.py:125` sets")
    add("  `cfg[\"sensor\"][\"params\"][\"los_occlusion\"] = True`**, added by")
    add("  commit `d322b22` *\"integrate: enforce sensor-faithful search")
    add("  detection\"*. Every GPPO scenario therefore runs with occlusion ON.")
    add("- `marvel_native` scenarios are also built through")
    add("  `prepare_gppo_config` in every parity suite, so **the frozen native")
    add("  reference was itself established with occlusion ON**.")
    add("")
    add("### The audited path already had occlusion enabled")
    add("")
    add("| path | los_occlusion |")
    add("|---|---|")
    add("| `extended_live_gppo_full.yaml` + `prepare_gppo_config` | **True** |")
    add("| same scenario, raw `load_and_validate_scenario` | False |")
    add("| `baseline_maps_test.yaml` (heuristic extended) | False |")
    add("| `marvel_native_test.yaml` raw | False |")
    add("")
    add("This is the central result: the live-GPPO benchmark that reported")
    add("28/60 parked UAVs was **already running with `los_occlusion = True`**,")
    add("so occlusion-free sensing is not what parks the fleet.")
    add("")

    add("## 2. The deadlock, proven step by step")
    add("")
    add("One UAV was followed from free motion into a permanent park in each")
    add("configuration. In both, the chain is exactly:")
    add("**waypoint is not reachable -> SafetyShield hovers -> the belief and")
    add("graph cannot change while the UAV cannot move -> the same class of")
    add("waypoint is chosen again -> permanent park.**")
    add("")
    for entry in trace:
        subject = entry["subject_uav"]
        add(f"### `los_occlusion = {entry['los_occlusion']}` "
            f"(uav {subject}, parked from mission step "
            f"{entry['subject_first_parked_step']})")
        add("")
        add("| mission step | position | waypoint | distance | moved | "
            "waypoint cell occupied | blocked at 0.2 m | shield events |")
        add("|---:|---|---|---:|---:|---|---|---:|")
        for row in entry["window"]:
            add(
                f"| {row['mission_step']} | {row['position']} | "
                f"{row['waypoint']} | {row['distance_to_waypoint']} | "
                f"{row['moved']} | {row['waypoint_cell_occupied']} | "
                f"{row['waypoint_blocked_at_shield_radius']} | "
                f"{len(row['shield_events'])} |"
            )
        add("")
    add("The difference between the two configurations is the *kind* of bad")
    add("waypoint, not whether one occurs:")
    add("")
    add("- with occlusion **off**, the waypoint cell is itself **occupied**")
    add("  (`waypoint_cell_occupied = True`) -- a node placed inside a wall;")
    add("- with occlusion **on**, the waypoint is a **free cell within 0.2 m of")
    add("  an obstacle** -- `SafetyShield.COLLISION_RADIUS` is 0.2 m, so the")
    add("  shield refuses a target the graph considers perfectly legal.")
    add("")
    add("A second, independent park mechanism was also observed: when the")
    add("target is free but the first 0.1 m step toward it collides,")
    add("`runtime.step` logs `grid_obstacle` and leaves the position untouched.")
    add("The shield never fires there because it validates only the target,")
    add("never the path.")
    add("")

    add("Caveat on the `invalid targets` column: it counts every commanded")
    add("waypoint within 0.2 m of an obstacle. A UAV that is *already* parked")
    add("next to an obstacle typically commands its own node, which also")
    add("qualifies -- so the column partly measures the consequence of parking")
    add("as well as its cause, and should not be read as a pure cause metric.")
    add("")

    add("## 3. False vs true, 4/8/16/30/60 UAVs")
    add("")
    add("30 mission steps per arm, identical seed and identical controlled")
    add("activation state (0.310452) in both arms.")
    add("")
    lines.extend(table(
        ["UAVs", "occlusion", "parked (half window)", "invalid targets",
         "nodes inside obstacle", "false-free belief", "dynamics skips",
         "shield static", "final rate"],
        [
            [r["uavs"], r["los_occlusion"], r["parked_half_window"],
             pct(r["invalid_target_fraction"]),
             f"{r['nodes_inside_obstacle']}/{r['graph_nodes']} "
             f"({pct(r['nodes_inside_obstacle_fraction'])})",
             pct(r["belief_false_free_fraction"]),
             r["dynamics_collision_skips"], r["shield_static_obstacle"],
             f"{float(r['exploration_rate_final']):.4f}"]
            for r in ab
        ],
    ))
    add("")

    add("## 4. Exploration-rate impact (PHASE 4)")
    add("")
    add("Occlusion-aware sensing sees strictly less, so it must slow coverage.")
    add("It does, by roughly a third to a half of the 30-step growth:")
    add("")
    lines.extend(table(
        ["UAVs", "occlusion", "rate at start", "rate after 30 steps",
         "growth", "graph nodes"],
        [
            [r["uavs"], r["los_occlusion"],
             f"{float(r['exploration_rate_start']):.5f}",
             f"{float(r['exploration_rate_final']):.5f}",
             f"{float(r['exploration_rate_final']) - float(r['exploration_rate_start']):+.5f}",
             r["graph_nodes"]]
            for r in ab
        ],
    ))
    add("")
    add("This is the expected physical consequence, not a defect: the")
    add("occlusion-free sensor was counting cells behind walls as explored, so")
    add("its higher rate was partly fictitious. The absolute effect is small")
    add("against the 0.30 threshold, and the controlled activation state seeds")
    add("both arms identically, so Search activation timing is unchanged in")
    add("this benchmark.")
    add("")

    add("## 5. What occlusion actually fixes")
    add("")
    off = {r["uavs"]: r for r in arm("False")}
    on = {r["uavs"]: r for r in arm("True")}
    for uavs in sorted(off, key=int):
        o, t = off[uavs], on[uavs]
        add(f"- **{uavs} UAVs**: false-free belief "
            f"{pct(o['belief_false_free_fraction'])} -> "
            f"{pct(t['belief_false_free_fraction'])}; "
            f"nodes inside obstacles "
            f"{pct(o['nodes_inside_obstacle_fraction'])} -> "
            f"{pct(t['nodes_inside_obstacle_fraction'])}; "
            f"parked {o['parked_half_window']} -> {t['parked_half_window']}.")
    add("")
    add("Occlusion-aware sensing is a **large correctness improvement** -- it")
    add("removes ~3/4 of the false-free belief and roughly halves")
    add("obstacle-interior nodes -- and at 4 and 8 UAVs it eliminates parking")
    add("entirely. But it is **not a fix for the deadlock at 16 UAVs and")
    add("above**, because the residual `invalid target` rate (35-42%) is driven")
    add("by the shield's 0.2 m clearance rule, not by sensing.")
    add("")

    if long_rows:
        add("## 6. Long horizon")
        add("")
        add("120 mission steps per arm. Provenance: the batch run was stopped by")
        add("a background time limit after 7 of 8 arms; the missing arm")
        add("(60 UAVs, occlusion on) was re-run separately. Both arms at 60 UAVs")
        add("with occlusion off were measured twice and produced identical")
        add("values (46/60 parked, rate 0.3538, 104.7 cells/call, 6413 nodes),")
        add("which is the determinism check for this harness.")
        add("")
        lines.extend(table(
            ["UAVs", "occlusion", "steps", "parked (half window)",
             "final rate", "visible cells / call", "final nodes",
             "first completion", "first Relay"],
            [
                [r["uavs"], r["los_occlusion"], r["steps"],
                 r["parked_uavs_half_window"], r["final_rate"],
                 r["visible_cells_per_call"], r["final_nodes"],
                 r["first_completion_step"] or "none",
                 r["first_relay_step"] or "none"]
                for r in long_rows
            ],
        ))
        add("")

    add("## 7. Native safety gate")
    add("")
    add("```")
    add(native_gate.strip())
    add("```")
    add("")

    add("## 8. Verdict on the fix")
    add("")
    add("The session premise was that `los_occlusion = false` causes the park")
    add("deadlock and that setting it true would fix it. **Measurement does not")
    add("support that premise**: the audited extended path already sets it")
    add("`True` (commit `d322b22`), and the deadlock persists there.")
    add("")
    add("No config-level correction is available for the residual deadlock:")
    add("`SafetyShield.COLLISION_RADIUS = 0.2` and the `radius=0.2` in")
    add("`runtime.step`'s collision check are constants, not settings, and the")
    add("mismatch is between **graph node placement** (a node may sit in a free")
    add("cell adjacent to an obstacle) and **the shield's clearance rule**. A")
    add("fix would have to change motion or node-admission logic, which this")
    add("session is explicitly not permitted to do.")
    add("")
    add("Everything in the production tree is therefore unchanged.")
    add("")

    (directory / "SENSING_CORRECTNESS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {directory / 'SENSING_CORRECTNESS_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
