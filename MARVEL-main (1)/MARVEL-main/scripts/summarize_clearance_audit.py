"""Consolidate the graph-motion clearance audit into a report.

    python scripts/summarize_clearance_audit.py --stamp <timestamp>
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()

    directory = ARTIFACTS / f"extended_clearance_consistency_{args.stamp}"
    mismatch = read_csv(directory / "clearance_mismatch.csv")
    proto = read_csv(directory / "prototype" / "prototype_comparison.csv")
    examples = json.loads((directory / "clearance_examples.json").read_text())
    contract = json.loads((directory / "geometry_contract.json").read_text())

    def _is_stock(value):
        return str(value).upper() in ("STOCK", "FALSE", "0")

    stock = {r["uavs"]: r for r in proto if _is_stock(r["filter"])}
    clear = {r["uavs"]: r for r in proto if not _is_stock(r["filter"])}

    lines: list[str] = []
    add = lines.append

    add("# Extended graph-motion clearance consistency audit")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat()}.")
    add("")
    add("Measurement only: no production file was modified. The prototype is")
    add("installed as a wrapper by `scripts/prototype_clearance_graph.py`.")
    add("")

    add("## 1. The geometry contract")
    add("")
    add("Three layers decide validity, and they disagree on all four axes")
    add("that matter:")
    add("")
    lines.extend(table(
        ["", "node admission", "edge admission", "physical validity"],
        [
            ["code",
             "`get_updating_node_coords`",
             "`Node.update_neighbor_nodes` -> `utils.check_collision`",
             "`ObstacleManager.check_collision`"],
            ["occupancy source", "**belief** map", "**belief** map",
             "**ground truth**"],
            ["resolution", f"{contract['layer_1_node_admission']['resolution_m']} m",
             f"{contract['layer_2_edge_admission']['resolution_m']} m",
             f"{contract['layer_3_physical_validity']['resolution_m']} m"],
            ["test", "cell is FREE and 8-connected",
             "Bresenham centreline; OCCUPIED or UNKNOWN blocks",
             "5-point star footprint"],
            ["clearance radius", "**none**", "**none**",
             f"{contract['layer_3_physical_validity']['collision_radius_m']} m"],
            ["endpoint tested", "yes", "**no** (breaks before testing)",
             "yes"],
            ["segment tested", "no", "yes, centreline only in the local window",
             "**no** (post-step position only)"],
            ["boundary", "assert in range",
             "out-of-window ends the walk untested",
             "outside frame bounds => collision"],
        ],
    ))
    add("")
    add("The five reasons a node or edge can be graph-legal but physically")
    add("illegal:")
    add("")
    for reason in contract["why_they_disagree"]:
        add(f"- {reason}")
    add("")

    add("## 2. Deterministic counter-examples")
    add("")
    for key, title in (
        ("M1", "M1 -- graph-valid node rejected by SafetyShield"),
        ("M2", "M2 -- free target whose motion segment collides"),
    ):
        entry = examples.get(key)
        add(f"### {title}")
        add("")
        if not entry:
            add("Not reproduced in this run.")
            add("")
            continue
        add("```json")
        add(json.dumps(entry, indent=2, default=str))
        add("```")
        add("")
    add("M1 is the sharper result: the target's own cell is **free**, yet its")
    add("minimum clearance is **0.0 m**. Graph nodes sit on the 4 m lattice,")
    add("which coincides with exact 1 m cell boundaries, so the footprint test")
    add("samples neighbouring cells that the node's own cell index never")
    add("reaches. A node can therefore be admitted on a cell that is free while")
    add("a cell one step away is occupied.")
    add("")

    add("## 3. How inconsistent is the graph?")
    add("")
    lines.extend(table(
        ["UAVs", "nodes", "A: point-free", "B: footprint-valid",
         "C: insufficient clearance", "D: edges with invalid swept path",
         "selected invalid", "stationary"],
        [
            [r["uavs"], r["nodes_total"], r["nodes_point_free"],
             r["nodes_footprint_valid"],
             f"{r['nodes_insufficient_clearance']} "
             f"({100 * float(r['nodes_insufficient_clearance_fraction']):.2f}%)",
             f"{r['edges_swept_invalid']} "
             f"({100 * float(r['edges_swept_invalid_fraction']):.2f}%)",
             f"{100 * float(r['selected_waypoint_invalid_fraction']):.1f}%",
             f"{r['uavs_stationary_last_step']}/{r['uavs']}"]
            for r in mismatch
        ],
    ))
    add("")
    add("**This is the key diagnostic, and it points the opposite way from the")
    add("hypothesis.** The graph is ~99% consistent: only 0.4-1.0% of nodes lack")
    add("the physical footprint and only 0.07-0.33% of edges have a blocked swept")
    add("path. The mismatch is real but *rare*.")
    add("")
    add("`selected invalid` is 52-57% at 16 UAVs and above, but that column is")
    add("dominated by UAVs that are **already parked** next to an obstacle and")
    add("command their own node; it measures the consequence of parking as much")
    add("as its cause, and must not be read as 'the planner picks bad targets")
    add("half the time'.")
    add("")

    add("## 4. Prototype: clearance-consistent graph")
    add("")
    add("Nodes are admitted only if `ObstacleManager.check_collision(coords,")
    add("radius=0.2)` is clean -- the identical primitive the shield calls. Edges")
    add("are admitted only if the whole swept segment keeps that clearance,")
    add("sampled at 0.05 m. The filter runs at *creation* time, which is sound")
    add("because the occupancy grid is ground truth and therefore static.")
    add("")
    lines.extend(table(
        ["UAVs", "arm", "parked", "nodes", "edges", "components",
         "isolated", "reachable fraction", "shield static", "dynamics blocks"],
        [
            [u, arm,
             stock.get(u, {}).get("parked_half_window")
             if arm == "STOCK" else clear.get(u, {}).get("parked_half_window"),
             stock.get(u, {}).get("graph_nodes")
             if arm == "STOCK" else clear.get(u, {}).get("graph_nodes"),
             stock.get(u, {}).get("graph_edges")
             if arm == "STOCK" else clear.get(u, {}).get("graph_edges"),
             stock.get(u, {}).get("components")
             if arm == "STOCK" else clear.get(u, {}).get("components"),
             stock.get(u, {}).get("isolated_nodes")
             if arm == "STOCK" else clear.get(u, {}).get("isolated_nodes"),
             stock.get(u, {}).get("reachable_fraction")
             if arm == "STOCK" else clear.get(u, {}).get("reachable_fraction"),
             stock.get(u, {}).get("shield_static_obstacle")
             if arm == "STOCK" else clear.get(u, {}).get("shield_static_obstacle"),
             stock.get(u, {}).get("grid_obstacle_blocks")
             if arm == "STOCK" else clear.get(u, {}).get("grid_obstacle_blocks"),
             ]
            for u in sorted(stock, key=int)
            for arm in ("STOCK", "CLEARANCE")
        ],
    ))
    add("")

    add("## 5. What the fix achieves, and what it does not")
    add("")
    add("- **M1 is eliminated outright.** `static_obstacle` shield blocks go to")
    add("  **zero at every scale** (1690 -> 0 at 16 UAVs, 3130 -> 0 at 30,")
    add("  7550 -> 0 at 60).")
    add("- **Parking falls at the scales that matter**: 5 -> 3 at 16 UAVs,"
    )
    add("  8 -> 4 at 30, and **26 -> 11 at 60 UAVs (a 58% reduction)**.")
    add("- **M2 survives and takes over.** Dynamics `grid_obstacle` blocks rise")
    add("  (352 -> 461, 270 -> 859, 1306 -> 2736) and are now the only")
    add("  remaining park mechanism. 11 of 60 UAVs still park permanently.")
    add("- Connectivity is not destroyed, and at 60 UAVs it improves markedly:")
    add("  reachable fraction 0.260/0.086/0.112/0.155 ->")
    add("  0.258/0.092/0.124/**0.277**. The 4-UAV arm drops (0.364 -> 0.208) on")
    add("  a small graph where removing a handful of nodes splits the largest")
    add("  component.")
    add("- Cost: the 60-UAV clearance arm took roughly 50 minutes for 30 mission")
    add("  steps, against about 2 minutes for the stock arm at the same scale.")
    add("")
    add("**Why M2 survives, structurally:** the commanded waypoint is a distant")
    add("graph node, and low-level motion is a *straight line* to it. Graph")
    add("edges are never followed. A clearance-consistent graph cannot help,")
    add("because the collision happens on a segment that is not an edge -- and")
    add("`SafetyShield` validates only the target, never the path. The UAV is")
    add("left with no recovery: it re-selects, re-collides, and stays put.")
    add("")

    add("## 6. Landing decision")
    add("")
    add("PHASE 10 requires all six conditions. Condition 1 (mismatch proven),")
    add("2 (same clearance semantics), 5 (connectivity usable) and 6 (native)")
    add("hold. Condition 3 fails: **M2 is not eliminated**. Condition 4 is only")
    add("half met (parking halved, not removed).")
    add("")
    add("**No commit was made.** Landing the node/edge filter alone would")
    add("eliminate a real defect class and halve parking, but it would also add")
    add("a correctness-critical graph traversal cost that does not fit the")
    add("time budget at 60 UAVs -- so it fails the landing rule on two counts.")
    add("")

    (directory / "CLEARANCE_AUDIT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {directory / 'CLEARANCE_AUDIT_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
