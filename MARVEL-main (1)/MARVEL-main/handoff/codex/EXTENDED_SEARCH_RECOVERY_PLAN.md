# Extended Search observable recovery implementation plan
> **For agentic workers:** Use executing-plans task-by-task. User authorized autonomous execution, no approval checkpoint or commits.
**Goal:** Safely connect actual extended UAV poses to observed graph Search routes.
**Architecture:** Publish physical occupancy only for already-visible cells; use GeometryFrame floor and the real five-probe footprint to gate segments. Reuse existing graph neighbors, persist safe recovery prefixes, invalidate rejected commands. When no whole waypoint is observed safe, clip a <=1m known-free prefix; a bounded temporary retreat may accompany desired_heading toward the public heat to expand FoV. This uses actual KinematicSimple control semantics and requires explicit model gating. Native routing stays unchanged.
**Tech Stack:** Python/NumPy/current runtime and SafetyShield.
**Spec:** artifacts/observable_search_recovery_20261009/OBSERVABLE_RECOVERY_COMPLETE.md

## Global constraints
- Existing extended-scalability checkout, no cleanup, training, commit or push.
- Frozen marvel_native unchanged. Radius 0.2m and SafetyShield retained.
- No truth-map input to planner; unknown is not certified free.
- New bounded validation only; don't repeat unchanged full simulations.
- Scope updated by user: technical goals only, no PPT/demo/technical report.

## Review focus
- Thin diagonal cell crossings: exact segment cell traversal.
- Missing or occluded labels: unknown fails closed, no truth leak.
- Partial movement then rejection: suppress repeated failed command.
- New assignment or reached prefix: reset stale recovery.
- Dynamic or stale geometry: keep shield/physical rejection, do not claim observed gate guarantees future safety.

### Task 1: Observation boundary
Files: utils/simulation_runtime.py; platform_tests/test_extended_search_recovery.py.
Interface: extended observation visible_occupancy = Nx3 col,row,0/1; occupancy_frame = frame.describe(); motion_feedback = own step/target/proposal/accepted/collision_type.
- [ ] Write tests for visible occupied target, occluded hidden cell, native unchanged, own feedback.
- [ ] Run red before runtime edit.
- [ ] Publish labels, record exact own proposal result per tick, clear on reset.
- [ ] Run green and native regressions.

### Task 2: Observed prefix controller
Files: integrations/gppo/search_recovery.py; same test file.
Interface: ObservedSearchRecovery(frame); observe(observations); select(uid,position,anchor,agent,preferred,base) -> waypoint.
- [ ] Test physical five probes, unknown, exact boundary traversal, actual-pose prefix vs graph edge, safe neighbor, rejection, persistence, assignment change.
- [ ] Run red before controller edit.
- [ ] Maintain only observed labels, gate preferred and base, then existing local graph neighbors <=6m; hold if none certified.
- [ ] Run green.

### Task 3: Extended integration
Files: integrations/gppo/scheduler.py; utils/policy_adapter.py.
- [ ] Integration red test: extended Search invokes controller, native bypasses.
- [ ] Feed observations from adapter; gate Search route and fallback, retain native behavior.
- [ ] Run complete platform suite and native parity vs baseline.
- [ ] Verify original observed snapshots with actual motion and shield.
- [ ] New bounded trace for recovery evidence; do not equate local success with mission completion.
- [ ] Independent review and one concrete repair pass; update technical acceptance ledger.

Self-review: all local feasibility design requirements covered, no global planner or frozen changes.
