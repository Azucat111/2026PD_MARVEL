"""Bounded validation of the NEW extended recovery implementation.
Snapshot inputs are re-sensed at saved fleet poses; this is not reconstruction
of missing observation history. Truth is used only by sensors/collision oracle.
"""
from pathlib import Path
import sys, json, time, csv, hashlib, subprocess
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
torch.set_num_threads(1)
from types import SimpleNamespace
from benchmark_live_gppo import build, SCENARIO
from utils.scenario_config import load_and_validate_scenario
from utils.obstacle_manager import ObstacleManager
from utils.sensor_models import IdealSensor
from utils.dynamics_models import create_dynamics_model
from utils.safety_shield import SafetyShield
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler

OUT = ROOT / "artifacts" / "extended_original_stall_snapshot_regression_20261010_attempt1"
OUT.mkdir(exist_ok=False)
config = load_and_validate_scenario(SCENARIO)
ob = ObstacleManager(config["environment"])
ob.load_from_file(config["environment"]["map_file"])
frame = ob.frame
sensor = IdealSensor({"fov":120, "range":10, "los_occlusion":True}, frame)
dyn = create_dynamics_model(config["dynamics"]["params"])
snapshot_results = []
for uid, mission in ((43,43), (0,56)):
    saved = ROOT / "artifacts/observable_search_recovery_20261009"
    meta = json.loads((saved / f"snapshot_{mission:03d}.json").read_text())
    arrays = np.load(saved / f"snapshot_{mission:03d}.npz")
    robots = [SimpleNamespace(**{**item, "position":np.array(item["position"])}) for item in meta["before"]]
    grid = ob.get_occupancy_grid()
    observations = {}
    positions = [(r.robot_id,r.position) for r in robots]
    for robot in robots:
        obs = sensor.sense(robot,grid,positions)
        obs["visible_occupancy"] = [[int(c),int(r),int(grid[r,c])] for c,r in obs["visible_cells"]]
        obs["occupancy_frame"] = frame.describe()
        obs["visible_occupancy_semantics"] = "areal_file_grid"
        observations[robot.robot_id] = obs
    previous_ticks = json.loads((saved / f"ticks_{mission-1:03d}.json").read_text())[str(uid)]
    for previous in reversed(previous_ticks):
        if not previous["collisions"] and np.array_equal(
                previous["after"]["position"], robots[uid].position):
            observations[uid]["motion_feedback"] = {
                "step":(mission-1)*10+previous["tick"], "accepted":True, "grid_probes_clear":True,
                "proposal":previous["after"]["position"],
                "target":meta["actions"][uid][0]}
            break
    controller = ObservedSearchRecovery(frame)
    controller.observe(observations)
    agent = SimpleNamespace(**{key:arrays[f"uav{uid}_{key}"] for key in
        ("node_coords","adjacent_matrix","neighbor_indices","current_index")})
    heat = meta["heat"][next(str(h) for h,u in meta["assignments"].items() if int(u)==uid)]
    robot = robots[uid]
    original = np.array(meta["actions"][uid][0])
    scheduler_probe = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler_probe.runtime = SimpleNamespace(obstacles=ob)
    scheduler_probe._search_recovery = controller
    target,heading = scheduler_probe._search_action(robot,heat,agent,(original,0),
        (np.asarray(meta["base_actions"][uid][0]),meta["base_actions"][uid][1]),route_goal_radius=3.5)
    selected = target.copy()
    if np.linalg.norm(target-robot.position) <= .1:
        snapshot_results.append({"uav_id":uid,"selected_target":target.tolist(),
            "status":"insufficient_reconstructed_visible_history", "legal_ticks":[],
            "heat_closing_m":0.})
        continue
    assert not controller.segment_safe(robot.position, original), "Unsafe original prefix unexpectedly certified"
    assert controller.segment_safe(robot.position, target)
    # Destroy oracle after observations: production choice must not change.
    saved_grid = ob._file_grid
    ob._file_grid = np.ones_like(saved_grid)
    again,_ = scheduler_probe._search_action(robot,heat,agent,(original,0),
        (np.asarray(meta["base_actions"][uid][0]),meta["base_actions"][uid][1]),route_goal_radius=3.5)
    np.testing.assert_array_equal(target,again)
    ob._file_grid = saved_grid
    start = robot.position.copy()
    commands = [(np.array(a),float(h)) for a,h in meta["actions"]]
    commands[uid] = (target,heading)
    ticks = []
    for tick in range(1,10*max(1,int(np.ceil(np.linalg.norm(target-start))))+1):
        filtered = SafetyShield(ob).filter_actions(robots, commands, tick)
        goal,angle = filtered[uid]
        state,_ = dyn.step(robot.position,goal,robot.heading,angle,robot.velocity,.1)
        proposed = state["position"]
        blocked,reason = ob.check_collision(proposed,radius=.2)
        assert not blocked, (uid,tick,reason)
        assert controller.segment_safe(robot.position,proposed)
        ticks.append({"tick":tick,"before":robot.position.tolist(),"after":proposed.tolist(),
            "collision":reason,"shield_target":np.asarray(goal).tolist()})
        robot.position=proposed
        robot.heading=state["heading"]

    assert np.linalg.norm(robot.position-target)<1e-9
    snapshot_results.append({"uav_id":uid,"mission":mission,"start":start.tolist(),
        "original_target":original.tolist(),"selected_target":selected.tolist(),
        "heading":heading,"legal_ticks":ticks,"oracle_independent":True,
        "observation_history":"re-sensed saved full fleet pose only",
        "heat_closing_m":float(np.linalg.norm(start-heat)-np.linalg.norm(target-heat))})
(OUT/"snapshot_validation.json").write_text(json.dumps(snapshot_results,indent=2))

print(json.dumps([{k:v for k,v in case.items() if k!="legal_ticks"} | {"legal_tick_count":len(case["legal_ticks"])} for case in snapshot_results]),flush=True)
