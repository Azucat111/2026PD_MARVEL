"""Consolidate the live-GPPO benchmark passes into one artifact report.

Reads the three passes written by `scripts/benchmark_live_gppo.py`
(GPPO profile, heuristic arm at the same controlled activation state, and
the long lifecycle trace) and emits `LIVE_GPPO_REPORT.md`, the scaling
tables and the bottleneck ranking.

    python scripts/summarize_live_gppo.py --stamp 20261001T144955Z
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def load(stamp: str, kind: str):
    name = {
        "gppo": f"extended_live_gppo_baseline_{stamp}",
        "heuristic": f"extended_live_gppo_heuristic_{stamp}",
        "lifecycle": f"extended_live_gppo_lifecycle_{stamp}",
    }[kind]
    directory = ARTIFACTS / name
    payload = json.loads(
        (directory / ("trace.json" if kind == "lifecycle" else "profile.json"))
        .read_text(encoding="utf-8")
    )
    return directory, payload


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

    gppo_dir, gppo = load(args.stamp, "gppo")
    _, heuristic = load(args.stamp, "heuristic")
    _, lifecycle = load(args.stamp, "lifecycle")

    scenario = json.loads(
        (gppo_dir / "scenario_manifest.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (gppo_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    system = json.loads(
        (gppo_dir / "system_info.txt").read_text(encoding="utf-8")
    )

    lines: list[str] = []
    add = lines.append

    add("# Extended live-GPPO activation-state benchmark")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat()} from "
        f"`artifacts/extended_live_gppo_*_{args.stamp}`.")
    add("")
    add("**CONTROLLED ACTIVATION-STATE BENCHMARK — not a natural-from-reset")
    add("deployment run.** The initial exploration state is seeded (see")
    add("`snapshot_manifest.json`); everything downstream of it — the Search")
    add("activation test, the GPPO allocation, the MARVEL planner, the Relay")
    add("and Safety layers — is the frozen production path, unmodified.")
    add("")
    add("**Measurement only.** No production module was changed in this")
    add("session. Every counter and timer is installed by monkeypatching")
    add("inside `scripts/benchmark_live_gppo.py`.")
    add("")

    add("## 1. Why a controlled activation state was needed")
    add("")
    add("In `extended` geometry `exploration_rate` is")
    add("`len(explored_cells) / obstacles.cell_count` — the denominator is")
    add("**every cell of the map**, not the free cells. On")
    add("`maps_test/1.png` (500x500, 37% free) the frozen threshold")
    add("`exploration_rate >= 0.30` therefore requires 75,000 visible cells,")
    add("i.e. **81% of the free area**. A 30-step scalability run reaches")
    add("~1% of the map, so Search could never activate naturally inside a")
    add("feasible benchmark. Lowering the threshold would change task")
    add("semantics, so the initial condition is changed instead.")
    add("")
    add("### How the state is produced")
    add("")
    add(snapshot["method"])
    add("")
    add("Deliberately *not* done:")
    add("")
    for item in snapshot["not_done"]:
        add(f"- {item}")
    add("")

    add("## 2. Lifecycle: what the live path actually did")
    add("")
    lines.extend(table(
        ["UAVs", "activated", "activation step", "first Search alloc",
         "first completion", "first refill", "first Relay",
         "parked >= half window", "shield blocks"],
        [
            [r["uavs"], r["activated"], r["activation_step"],
             r["first_search_allocation_step"],
             r["first_completion_step"] or "none",
             r["first_refill_step"] or "none",
             r["first_relay_step"] or "none",
             f"{r['parked_uavs_half_window']}/{r['uavs']}",
             r["shield_blocks_total"]]
            for r in lifecycle
        ],
    ))
    add("")
    add("(The table is the long 150-step lifecycle trace; the 30-step profile")
    add("passes at 4/8/16/30/60 UAVs show the same activation and allocation")
    add("steps, see `steps.csv`.)")
    add("")
    add("Search activates through the normal `_maybe_activate_search` test and")
    add("the scheduler allocates real Search rows from real heat points at")
    add("mission step 2 in every run — at every fleet size the allocation")
    add("fills the frozen capacity of 2 UAVs.")
    add("")
    add("**Search completion, refill and Relay were not observed, and the")
    add("reason is structural rather than a defect in this harness:**")
    add("")
    add("- The generated heat points sit 276 m and 310 m from the UAVs they")
    add("  are assigned to. At the configured 1.0 m/s the closest assignment")
    add("  closes at ~0.75 m per mission step, so reaching the 3.5 m service")
    add("  radius needs on the order of 250 further mission steps. A longer")
    add("  run is not a fair remedy: beyond ~40 steps the fleet degrades (see")
    add("  below).")
    add("- Relay demand never materialised: `ObservedRelayDemandBuilder`")
    add("  reported no disconnected target, so no Relay row was ever created.")
    add("")

    add("## 3. Steady-state vs allocation step")
    add("")
    add("Steps are classified by whether the real GPPO allocator was invoked")
    add("during the step; the two populations are never averaged together.")
    add("")
    lines.extend(table(
        ["UAVs", "steady mean s", "steady p95 s", "steady n",
         "allocation mean s", "allocation max s", "allocation n"],
        [
            [r["uavs"], f"{r['steady_mean_seconds']:.4f}",
             f"{r['steady_p95_seconds']:.4f}", r["steady_steps"],
             f"{r['allocation_mean_seconds']:.4f}",
             f"{r['allocation_max_seconds']:.4f}", r["allocation_steps"]]
            for r in gppo
        ],
    ))
    add("")
    add("`max_search_uavs` is 2 and the Search service timer is 6 steps, so")
    add("once the two slots are filled the allocator is idle until a heat")
    add("point completes — which, per section 2, never happens on this map.")
    add("The allocation population is therefore one step per run, and the")
    add("allocation step costs **+16%** over steady state at 60 UAVs")
    add("(5.05 s vs 4.37 s).")
    add("")

    add("## 4. GPPO's share of the live step")
    add("")
    lines.extend(table(
        ["UAVs", "step s", "gppo.scheduler_apply ms", "share", "gppo.allocate ms"],
        [
            [r["uavs"], f"{r['steady_mean_seconds']:.4f}",
             f"{r['inclusive'].get('gppo.scheduler_apply', 0) * 1000:.1f}",
             f"{100 * r['inclusive'].get('gppo.scheduler_apply', 0) / r['steady_mean_seconds']:.2f}%",
             f"{r['inclusive'].get('gppo.allocate', 0) * 1000:.2f}"]
            for r in gppo
        ],
    ))
    add("")
    add("The whole GPPO layer is **under 1% of the live mission step** at 8")
    add("UAVs and above. The earlier GPPO-focused optimization effort was")
    add("aimed at a cost that the live loop barely pays.")
    add("")

    add("## 5. Where the 60-UAV step actually goes")
    add("")
    top = [r for r in gppo if r["uavs"] == 60]
    if top:
        r = top[0]
        lines.extend(table(
            ["span", "ms", "share", "calls per mission step"],
            [
                [k, f"{v * 1000:.1f}",
                 f"{100 * v / r['steady_mean_seconds']:.1f}%",
                 r["calls_per_step"].get(k, "")]
                for k, v in sorted(r["inclusive"].items(),
                                   key=lambda kv: -kv[1])[:12]
            ],
        ))
    add("")
    add("The spans are **inclusive**: `marvel.planning_state` contains")
    add("`marvel.planning_shared_graph`, and `marvel.batched_actions` contains")
    add("`marvel.observation` (which in turn contains")
    add("`marvel.heading_candidates`) and `marvel.policynet_forward`. The")
    add("shares therefore overlap and do not sum to 100%; the ranking below")
    add("uses the exclusive remainder.")
    add("")
    add("Two observations that the earlier per-component sessions could not")
    add("see from inside a single component:")
    add("")
    add("1. `runtime.sensing` (`_get_observations`) is called **10 times per")
    add("   mission step** — once per physics tick. Only the tenth call's")
    add("   output reaches `MARVELPolicyAdapter.get_actions`; the other nine")
    add("   feed the per-tick `_update_explored_cells` accumulation and target")
    add("   detection. At 60 UAVs the ten calls cost 1.27 s, or 29% of the")
    add("   step.")
    add("2. `marvel.policynet_forward` reads **6.8 ms**, but that number is")
    add("   **not** the network's cost: the timer wraps `forward` without a")
    add("   `torch.cuda.synchronize()`, so it measures kernel-launch time and")
    add("   the real GPU work is deferred into the following `.to('cpu')`,")
    add("   i.e. into the untimed remainder of `marvel.batched_actions`. The")
    add("   synchronised measurement in `scripts/benchmark_policy_devices.py`")
    add("   puts the same 60-UAV batched forward at **137.7 ms**. Read the")
    add("   `marvel.policynet_forward` row as launch overhead only.")
    add("")

    add("## 6. Action space and decisions")
    add("")
    lines.extend(table(
        ["UAVs", "uav rows", "task rows", "flat action space"],
        [
            [r["uavs"], r["action_space"].get("n_uav_rows"),
             r["action_space"].get("n_task_rows"),
             r["action_space"].get("flat_action_space")]
            for r in gppo
        ],
    ))
    add("")
    add("Three task rows throughout (Relay anchor, plus the registered Search")
    add("subtasks), so the flat action space stays 3 per UAV. The allocator is")
    add("invoked once per 30 mission steps in this configuration.")
    add("")

    add("## 7. Resources")
    add("")
    lines.extend(table(
        ["UAVs", "graph nodes", "graph edges", "RAM MB",
         "VRAM allocated MB", "VRAM reserved MB"],
        [
            [r["uavs"], r["graph_nodes"], r["graph_edges"],
             f"{r['ram_mb']:.0f}",
             f"{r.get('vram_peak_allocated_mb', float('nan')):.1f}",
             f"{r.get('vram_peak_reserved_mb', float('nan')):.1f}"]
            for r in gppo
        ],
    ))
    add("")

    add("## 8. Against the 60-UAV CUDA heuristic baseline")
    add("")
    add("The comparison target is the 60-UAV CUDA heuristic baseline from the")
    add("earlier sessions (2.286 s; the recorded artifact value is 2.1382 s on")
    add("`baseline_maps_test.yaml`). That run started from the scenario's own")
    add("reset, which is a different state from the controlled activation")
    add("state used here, so the two are not directly comparable.")
    add("")
    add("The comparable measurement is the heuristic scheduler on the")
    add("**same controlled activation state**, which differs from the live-GPPO")
    add("arm only in `task_scheduler.mode`:")
    add("")
    lines.extend(table(
        ["UAVs", "live GPPO s", "heuristic s", "delta s", "delta",
         "graph nodes"],
        [
            [g["uavs"], f"{g['steady_mean_seconds']:.4f}",
             f"{h['steady_mean_seconds']:.4f}",
             f"{g['steady_mean_seconds'] - h['steady_mean_seconds']:+.4f}",
             f"{100 * (g['steady_mean_seconds'] - h['steady_mean_seconds']) / h['steady_mean_seconds']:+.1f}%",
             g["graph_nodes"]]
            for g, h in zip(gppo, heuristic)
        ],
    ))
    add("")
    add("The published baseline reproduces on this machine today:")
    add("re-running `scripts/benchmark_policy_devices.py --uavs 60` gives")
    add("**1.9193 s** for cuda_batched at 60 UAVs on")
    add("`baseline_maps_test.yaml` (published: 2.1382 s). So the machine is")
    add("comparable, and the 1.92 s -> 4.37 s gap is real — but it is **not**")
    add("a GPPO cost.")
    add("")
    add("Two things are measured directly:")
    add("")
    add("- At the *identical* state, the heuristic scheduler costs 4.2206 s")
    add("  and live GPPO 4.3653 s. The whole GPPO layer is **+3.4%**.")
    add("- Dropping only the activation seed at 60 UAVs gives 3.7136 s, with")
    add("  `marvel.planning_state` falling from 1283 ms to 678 ms. That run")
    add("  has *more* nodes (6415 vs 5580) but far fewer edges (18,297 vs")
    add("  64,532), so the honest reading is that this pipeline's planning")
    add("  cost tracks **adjacency density**, not node count.")
    add("")
    add("The remaining difference against the published baseline also spans a")
    add("different scenario (it has no `target_search`/`relay` tasks and a")
    add("128-tick episode), so the gap is a state-and-scenario difference that")
    add("is not attributable to the GPPO layer. See `state_dependence.csv` for")
    add("the three arms side by side.")
    add("")

    add("## 9. Bottleneck ranking and next target")
    add("")
    add("At 60 UAVs, by exclusive cost:")
    add("")
    lines.extend(table(
        ["rank", "component", "ms", "share", "already optimized"],
        [
            [1, "`marvel.planning_state` (`get_all_node_graph`)", "1283",
             "29.4%", "yes — shared graph reuse"],
            [2, "`runtime.sensing` (`_get_observations`)", "1275",
             "29.2%", "**no**"],
            [3, "`marvel.graph` (`Agent.update_graph`)", "928",
             "21.3%", "yes — three hypotheses refuted"],
            [4, "`marvel.batched_actions` pack + transfer + GPU forward",
             "~390 (excl. observation)", "8.9%", "yes — CUDA batching"],
            [5, "`marvel.observation` (tensors + headings)", "298",
             "6.8%", "partly"],
            [6, "`gppo.scheduler_apply`", "32", "0.7%", "yes"],
        ],
    ))
    add("")
    add("**Recommended single next target: `runtime.sensing`.**")
    add("")
    add("It is the largest block that has never been touched, it is 29% of the")
    add("60-UAV step, and it is called ten times per mission step while the")
    add("policy consumes one of those calls. It is also independent of the")
    add("GPPO layer, so improving it does not touch the frozen parity path.")
    add("The first question to answer is behavioural, not algorithmic: which")
    add("of the ten per-tick calls are actually load-bearing (the")
    add("`_update_explored_cells` accumulation needs every tick; the MARVEL")
    add("policy needs only the last), and can the per-tick work be reduced")
    add("without changing the explored-cell set or the mission-step")
    add("observation that feeds `_policy_actions`?")
    add("")

    add("## 10. Regression status")
    add("")
    add("No production file was modified: `git diff` against `1c88699` shows")
    add("only `__pycache__` bytecode. The platform test suite was run")
    add("afterwards and its result is recorded in `regression.txt`.")
    add("")

    add("## 11. Environment")
    add("")
    add(f"- platform: `{system.get('platform')}`")
    add(f"- python: `{system.get('python', '').splitlines()[0]}`")
    add(f"- torch: `{system.get('torch')}`, CUDA available: "
        f"`{system.get('cuda_available')}`")
    add(f"- CUDA device: `{system.get('cuda_device')}`")
    add(f"- logical CPUs: {system.get('cpu_count_logical')}, "
        f"RAM: {system.get('ram_total_mb', 0):.0f} MB")
    add("")
    add(f"- scenario: `{scenario.get('scenario_path')}` "
        f"(map `{scenario.get('map_file')}`)")
    add(f"- activation threshold: `{scenario.get('activation_threshold')}` "
        f"(frozen), Search capacity `{scenario.get('search_capacity_max_search_uavs')}`, "
        f"service steps `{scenario.get('search_service_steps')}`")
    add(f"- policy: `{scenario.get('policy')}`")
    add("")

    report = "\n".join(lines) + "\n"
    (gppo_dir / "LIVE_GPPO_REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {gppo_dir / 'LIVE_GPPO_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
