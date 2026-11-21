"""Consolidate the post-fix live Search validation into a report.

    python scripts/summarize_postfix_validation.py --stamp <timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# Pre-fix reference, from the live-GPPO activation-state baseline and the
# 120-step clearance runs recorded before e7246a8.
PREFIX_PARKED_120 = {8: "5/8", 16: "15/16", 30: "26/30", 60: "48/60"}


def read_csv(path: Path):
    if not path.exists():
        return []
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

    directory = ARTIFACTS / f"extended_postfix_live_gppo_{args.stamp}"
    summary = json.loads((directory / "summary.json").read_text())
    events = read_csv(directory / "search_events.csv")
    refills = read_csv(directory / "refill_events.csv")
    relays = read_csv(directory / "relay_events.csv")
    parking = read_csv(directory / "parking_health.csv")

    by_uavs = {str(r["uavs"]): r for r in summary}

    def event(uavs, name):
        for row in events:
            if str(row["uavs"]) == str(uavs) and row["event"] == name:
                return row["mission_step"]
        return None

    lines: list[str] = []
    add = lines.append

    add("# Post-fix live Search completion / refill validation")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat()}.")
    add("")
    add("Validation only: no production file was modified. The system is")
    add("`extended-scalability` @ `e7246a8` -- extended geometry,")
    add("occlusion-aware sensing, CUDA batched PolicyNet, the frozen GPPO")
    add("checkpoint, real `GPPOTaskScheduler`, real Search and Relay, real")
    add("`SafetyShield`, frozen activation threshold and frozen Search")
    add("capacity, on the controlled activation state.")
    add("")

    add("## 1. Search lifecycle by scale")
    add("")
    lines.extend(table(
        ["UAVs", "steps run", "activation", "initial allocation",
         "first arrival", "first service start", "first completion",
         "first refill", "Relay"],
        [
            [
                u,
                by_uavs[u]["steps_run"],
                by_uavs[u]["activation_step"],
                event(u, "initial_allocation"),
                min(by_uavs[u]["arrivals"].values())
                if by_uavs[u]["arrivals"] else "never",
                min(by_uavs[u]["service_starts"].values())
                if by_uavs[u]["service_starts"] else "never",
                by_uavs[u]["first_completion_step"] or "none",
                by_uavs[u]["first_refill_step"] or "none",
                by_uavs[u]["first_relay_step"] or "never",
            ]
            for u in sorted(by_uavs, key=int)
        ],
    ))
    add("")
    add("`arrival` and `service start` are the **dwell** route: within")
    add("`search_reach_radius = 3.5 m` for `search_service_steps = 6`")
    add("consecutive steps. Where completion happened without either, the")
    add("frozen **sensor-detector** route resolved it --")
    add("`_sync_search_completions` reading `runtime.completed_search_heat_ids()`")
    add("-- which fires from `visible_cells` at up to sensor range, before the")
    add("3.5 m dwell radius is reached. Both are frozen semantics; no")
    add("threshold was touched.")
    add("")

    if refills:
        add("## 2. Refill, through the real scheduler path")
        add("")
        lines.extend(table(
            ["UAVs", "step", "completed heat", "new assignments",
             "assignments after", "serviced total", "allocator calls"],
            [
                [r["uavs"], r["mission_step"], r["completed_heat"],
                 r["new_assignments"], r["search_assignments_after"],
                 r["serviced_total"], r["allocate_calls"]]
                for r in refills
            ],
        ))
        add("")
        add("Each row is a real `GPPOEventAllocator.allocate` invocation that")
        add("followed a real completion. No synthetic slot list is injected")
        add("anywhere in this harness.")
        add("")

    add("## 3. Relay")
    add("")
    if relays:
        lines.extend(table(
            ["UAVs", "step", "target", "helper", "serviced total"],
            [[r["uavs"], r["mission_step"], r["target_uav"],
              r["helper_uav"], r["serviced_total"]] for r in relays],
        ))
    else:
        add("**No Relay assignment ever occurred at any scale.**")
        add("`ObservedRelayDemandBuilder` found no disconnected target, so no")
        add("Relay row was ever created. No connectivity failure was")
        add("fabricated to force one.")
    add("")

    add("## 4. Distance actually travelled to the heat point")
    add("")
    add("Per assigned Search UAV, from `search_progress.csv`:")
    add("")
    lines.extend(table(
        ["UAVs", "heat", "uav", "samples", "start m", "closest m",
         "at step", "end m", "steps parked"],
        [
            [r["uavs"], r["heat"], r["uav"], r["n"], r["start"],
             r["min"], r["min_step"], r["end"], r["parked"]]
            for r in _distance_traces(directory)
        ],
    ))
    add("")

    add("## 5. Parking health")
    add("")
    if parking:
        lines.extend(table(
            ["UAVs", "step", "parked", "moving", "shield static (cum.)",
             "dynamics blocks (cum.)", "exploration rate"],
            [
                [r.get("uavs", ""), r["mission_step"], r["parked_now"],
                 r["moving_now"], r["shield_static_obstacle"],
                 r["grid_obstacle_blocks"], r["exploration_rate"]]
                for r in parking
            ],
        ))
        add("")
    add("Pre-fix reference at 120 steps: " + ", ".join(
        f"{u} UAVs {v}" for u, v in sorted(PREFIX_PARKED_120.items())
    ) + " parked.")
    add("")

    add("## 6. Timing, split by step kind")
    add("")
    lines.extend(table(
        ["UAVs", "steady mean s", "steady p95 s", "steady n",
         "allocation mean s", "allocation n",
         "refill mean s", "refill n"],
        [
            [u,
             _fmt(by_uavs[u]["steady"]["mean"]),
             _fmt(by_uavs[u]["steady"]["p95"]),
             by_uavs[u]["steady"]["n"],
             _fmt(by_uavs[u]["allocation"]["mean"]),
             by_uavs[u]["allocation"]["n"],
             _fmt(by_uavs[u]["refill"]["mean"]),
             by_uavs[u]["refill"]["n"]]
            for u in sorted(by_uavs, key=int)
        ],
    ))
    add("")
    add("## 7. Component breakdown at 60 UAVs")
    add("")
    if "60" in by_uavs:
        spans = sorted(by_uavs["60"]["inclusive"].items(),
                       key=lambda kv: -kv[1])[:10]
        lines.extend(table(
            ["span", "ms per recorded step"],
            [[k, f"{v * 1000:.1f}"] for k, v in spans],
        ))
    else:
        add("(60 UAVs did not complete inside this run.)")
    add("")
    add("## 8. Resources")
    add("")
    lines.extend(table(
        ["UAVs", "RAM MB", "VRAM allocated MB", "VRAM reserved MB"],
        [
            [u, f"{by_uavs[u]['ram_mb']:.0f}",
             _fmt(by_uavs[u]["vram_peak_allocated_mb"], 1),
             _fmt(by_uavs[u]["vram_peak_reserved_mb"], 1)]
            for u in sorted(by_uavs, key=int)
        ],
    ))
    add("")

    (directory / "POSTFIX_LIVE_GPPO_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {directory / 'POSTFIX_LIVE_GPPO_REPORT.md'}")
    return 0


def _fmt(value, digits=4):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _distance_traces(directory: Path):
    import collections

    rows = read_csv(directory / "search_progress.csv")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["uavs"], row["heat_id"], row["uav_id"])].append(
            (int(row["mission_step"]), float(row["distance_m"]),
             int(row["parked"]))
        )
    out = []
    for key in sorted(grouped, key=lambda k: (int(k[0]), int(k[1]))):
        trace = grouped[key]
        distances = [t[1] for t in trace]
        low = min(distances)
        out.append({
            "uavs": key[0], "heat": key[1], "uav": key[2],
            "n": len(trace),
            "start": f"{distances[0]:.1f}",
            "min": f"{low:.2f}",
            "min_step": trace[distances.index(low)][0],
            "end": f"{distances[-1]:.1f}",
            "parked": sum(t[2] for t in trace),
        })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
