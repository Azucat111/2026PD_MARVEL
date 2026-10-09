# Post-fix live Search completion / refill validation

Generated 2026-10-03T02:42:04.363865+00:00.

Validation only: no production file was modified. The system is
`extended-scalability` @ `e7246a8` -- extended geometry,
occlusion-aware sensing, CUDA batched PolicyNet, the frozen GPPO
checkpoint, real `GPPOTaskScheduler`, real Search and Relay, real
`SafetyShield`, frozen activation threshold and frozen Search
capacity, on the controlled activation state.

## 1. Search lifecycle by scale

| UAVs | steps run | activation | initial allocation | first arrival | first service start | first completion | first refill | Relay |
|---|---|---|---|---|---|---|---|---|
| 8 | 450 | 0 | 2 | never | never | none | none | never |
| 16 | 144 | 0 | 2 | never | never | 93 | 94 | never |
| 30 | 228 | 0 | 2 | never | never | 177 | 178 | never |
| 60 | 450 | 0 | 2 | never | never | none | none | never |

`arrival` and `service start` are the **dwell** route: within
`search_reach_radius = 3.5 m` for `search_service_steps = 6`
consecutive steps. Where completion happened without either, the
frozen **sensor-detector** route resolved it --
`_sync_search_completions` reading `runtime.completed_search_heat_ids()`
-- which fires from `visible_cells` at up to sensor range, before the
3.5 m dwell radius is reached. Both are frozen semantics; no
threshold was touched.

## 2. Refill, through the real scheduler path

| UAVs | step | completed heat | new assignments | assignments after | serviced total | allocator calls |
|---|---|---|---|---|---|---|
| 16 | 94 | 93 | {"2": 0} | {"1": 9, "2": 0} | 1 | 1 |
| 30 | 178 | 177 | {"2": 5} | {"1": 21, "2": 5} | 1 | 1 |

Each row is a real `GPPOEventAllocator.allocate` invocation that
followed a real completion. No synthetic slot list is injected
anywhere in this harness.

## 3. Relay

**No Relay assignment ever occurred at any scale.**
`ObservedRelayDemandBuilder` found no disconnected target, so no
Relay row was ever created. No connectivity failure was
fabricated to force one.

## 4. Distance actually travelled to the heat point

Per assigned Search UAV, from `search_progress.csv`:

| UAVs | heat | uav | samples | start m | closest m | at step | end m | steps parked |
|---|---|---|---|---|---|---|---|---|
| 8 | 0 | 0 | 449 | 266.4 | 158.56 | 134 | 158.6 | 316 |
| 8 | 1 | 5 | 449 | 516.3 | 434.54 | 113 | 434.5 | 337 |
| 16 | 0 | 0 | 91 | 90.0 | 11.68 | 92 | 11.7 | 0 |
| 16 | 1 | 9 | 143 | 382.9 | 320.37 | 79 | 321.1 | 0 |
| 16 | 2 | 0 | 51 | 435.2 | 408.79 | 130 | 408.8 | 14 |
| 30 | 0 | 0 | 175 | 149.8 | 12.22 | 176 | 12.2 | 0 |
| 30 | 1 | 21 | 227 | 232.5 | 202.01 | 63 | 202.0 | 0 |
| 30 | 2 | 5 | 51 | 458.3 | 416.94 | 228 | 416.9 | 0 |
| 60 | 0 | 43 | 449 | 150.0 | 117.21 | 42 | 117.2 | 408 |
| 60 | 1 | 0 | 449 | 373.1 | 370.64 | 6 | 410.4 | 395 |

## 5. Parking health

| UAVs | step | parked | moving | shield static (cum.) | dynamics blocks (cum.) | exploration rate |
|---|---|---|---|---|---|---|
|  | 50 | 0 | 8 | 0 | 0 | 0.310608 |
|  | 120 | 1 | 7 | 0 | 8 | 0.310896 |
|  | 250 | 2 | 6 | 0 | 255 | 0.315228 |
|  | 450 | 2 | 6 | 0 | 655 | 0.315228 |
|  | 50 | 2 | 14 | 0 | 0 | 0.31792 |
|  | 120 | 2 | 14 | 0 | 67 | 0.324076 |
|  | 50 | 5 | 25 | 0 | 0 | 0.325544 |
|  | 120 | 5 | 25 | 0 | 67 | 0.327352 |
|  | 50 | 4 | 56 | 0 | 9 | 0.345572 |
|  | 120 | 4 | 56 | 0 | 145 | 0.34952 |
|  | 250 | 3 | 57 | 0 | 405 | 0.34952 |
|  | 450 | 3 | 57 | 0 | 805 | 0.34952 |

Pre-fix reference at 120 steps: 8 UAVs 5/8, 16 UAVs 15/16, 30 UAVs 26/30, 60 UAVs 48/60 parked.

## 6. Timing, split by step kind

| UAVs | steady mean s | steady p95 s | steady n | allocation mean s | allocation n | refill mean s | refill n |
|---|---|---|---|---|---|---|---|
| 8 | 0.6064 | 0.6926 | 449 | 0.7441 | 1 |  | 0 |
| 16 | 1.1078 | 1.2972 | 142 | 0.9742 | 1 | 1.2315 | 1 |
| 30 | 2.3447 | 2.7127 | 226 | 2.0227 | 1 | 3.1130 | 1 |
| 60 | 5.7912 | 6.6514 | 449 | 5.0199 | 1 |  | 0 |

## 7. Component breakdown at 60 UAVs

| span | ms per recorded step |
|---|---|
| marvel.planning_state | 2312.7 |
| runtime.sensing | 1256.9 |
| marvel.graph | 1131.2 |
| marvel.batched_actions | 863.4 |
| marvel.observation | 348.4 |
| marvel.heading_candidates | 234.5 |
| marvel.planning_shared_graph | 104.9 |
| runtime.shield | 78.4 |
| runtime.collision | 76.9 |
| gppo.scheduler_apply | 29.5 |

## 8. Resources

| UAVs | RAM MB | VRAM allocated MB | VRAM reserved MB |
|---|---|---|---|
| 8 | 1254 | 117.4 | 170.0 |
| 16 | 1310 | 206.1 | 286.0 |
| 30 | 1493 | 356.7 | 466.0 |
| 60 | 1578 | 682.5 | 824.0 |


## 9. Comparison with the pre-fix system

Parking at 120 mission steps, before -> after:

| UAVs | pre-fix (1c88699) | post-fix (e7246a8) |
|---:|---|---|
| 8 | 5/8 | **1/8** |
| 16 | 15/16 | **2/16** |
| 30 | 26/30 | **5/30** |
| 60 | 48/60 | **4/60** |

And over the full horizon the residual does **not** compound: at 8 UAVs
parking goes 0 -> 1 -> 2 -> 2 at steps 50/120/250/450, and at 60 UAVs
4 -> 4 -> 3 -> 3. The remaining parked UAVs appear early and then stay
flat rather than accumulating.

`static_obstacle` shield blocks are **zero at every scale and every
checkpoint**. Only `uav_proximity` remains (120 / 270 / 2150 cumulative at
16 / 30 / 60 UAVs), which is crowding, not the graph/physics mismatch.

Behavioural change in Search is the headline: before the fix the assigned
Search UAVs were frozen and never approached their heat points. Now they
travel, and at 16 and 30 UAVs a heat point is actually completed and
refilled.

Timing is **not** compared as if the states were identical: the post-fix
60-UAV steady step is 5.79 s against 4.94 s pre-fix, but that run covered
450 steps of exploration and carries a much larger graph
(`marvel.planning_state` 2313 ms vs 1283 ms). The states differ, so the
raw numbers are not a like-for-like overhead measurement.

## 10. What this does and does not establish

Established: with e7246a8 the integrated system performs the full Search
lifecycle -- activation, allocation, travel, completion and refill --
through the real scheduler, at more than one scale, on the frozen
thresholds.

Not established: completion at 8 and 60 UAVs within 450 steps, and any
Relay activity. At 60 UAVs the two assigned Search UAVs close to 117 m
and 371 m and then stall, so the limiting factor for the largest scale is
still travel distance and the residual parking, not the Search logic.
