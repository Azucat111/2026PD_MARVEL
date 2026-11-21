"""Extended observed physical occupancy boundary."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.geometry import GeometryFrame
from utils.sensor_models import IdealSensor
from utils.simulation_runtime import RobotState, SimulationRuntime

def observation_runtime(mode="extended"):
    frame = GeometryFrame(0, 0, 1, 12, 12, mode=mode)
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[5, 5] = 1
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    runtime.geometry_mode = mode
    runtime.obstacles = SimpleNamespace(get_occupancy_grid=lambda: grid.copy(), frame=frame)
    runtime.robots = [RobotState(0, "uav", np.array([3., 5.]), 1., 0.)]
    runtime.sensor = IdealSensor({"fov": 120, "range": 8, "los_occlusion": True}, frame)
    runtime.sensor_by_type = {}
    runtime._motion_feedback = {}
    return runtime, grid

def test_extended_labels_are_only_already_visible_cells():
    runtime, grid = observation_runtime()
    observation = runtime._get_observations()[0]
    labels = np.asarray(observation["visible_occupancy"])
    assert set(map(tuple, labels[:, :2])) == set(map(tuple, observation["visible_cells"]))
    assert [5, 5, 1] in labels.tolist()
    assert [7, 5] not in labels[:, :2].tolist()
    for col, row, value in labels:
        assert value == grid[row, col]
    assert observation["occupancy_frame"]["cell_size"] == 1.

def test_native_observation_has_no_extended_recovery_fields():
    runtime, _ = observation_runtime("marvel_native")
    observation = runtime._get_observations()[0]
    assert not {"visible_occupancy", "occupancy_frame", "motion_feedback"} & set(observation)

def test_feedback_publishes_only_own_motion_result():
    runtime, _ = observation_runtime()
    runtime._motion_feedback = {
        0: {"step": 3, "accepted": False, "target": [8., 5.], "proposal": [4.9, 5.],
            "collision_type": "grid_obstacle"},
        99: {"secret": "another UAV"},
    }
    observation = runtime._get_observations()[0]
    assert observation["motion_feedback"] == runtime._motion_feedback[0]
    assert "secret" not in observation["motion_feedback"]

from integrations.gppo.search_recovery import ObservedSearchRecovery

def recovery(grid=None, frame=None):
    frame = frame or GeometryFrame.extended(width_cells=12, height_cells=12)
    controller = ObservedSearchRecovery(frame)
    if grid is not None:
        controller.observe({0: {"visible_occupancy":
            [[c, r, int(grid[r, c])] for r in range(grid.shape[0]) for c in range(grid.shape[1])],
            "occupancy_frame": frame.describe(), "motion_feedback": {}}})
    return controller

def test_unknown_cells_cannot_certify_prefix():
    assert not recovery().segment_safe([2., 2.], [3., 2.])

def test_physical_radius_blocks_free_centerline():
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[5, 5] = 1
    controller = recovery(grid)
    assert not controller.segment_safe([4.5, 4.9], [6., 4.9])
    assert controller.segment_safe([4.5, 4.7], [6., 4.7])

def test_exact_traversal_rejects_tiny_corner_crossing():
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[1, 2] = 1
    assert not recovery(grid).segment_safe([1.95, 1.95], [2.05, 2.0499998], radius=0)

def graph():
    return SimpleNamespace(node_coords=np.array([[4., 4.], [6., 5.], [4., 8.], [4., 2.]]),
        adjacent_matrix=np.zeros((4, 4)), current_index=0,
        neighbor_indices=np.arange(4))

def blocked_controller():
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[5, 5] = 1
    return recovery(grid)

def test_actual_position_gate_selects_existing_legal_neighbor():
    controller = blocked_controller()
    waypoint = controller.select(43, [4.4, 4.8], [10., 8.], graph(), [6., 5.], [8., 5.])
    np.testing.assert_array_equal(waypoint, [4., 8.])
    assert controller.segment_safe([4.4, 4.8], waypoint)

def test_recovery_prefix_persists_until_reached():
    controller = blocked_controller()
    controller.select(43, [4.4, 4.8], [10., 8.], graph(), [6., 5.], [8., 5.])
    waypoint = controller.select(43, [4.3, 5.3], [10., 8.], graph(), [4., 2.], [4., 2.])
    np.testing.assert_array_equal(waypoint, [4., 8.])

def test_rejected_prefix_is_not_reissued():
    controller = blocked_controller()
    controller.select(43, [4.4, 4.8], [10., 8.], graph(), [6., 5.], [8., 5.])
    controller.observe({43: {"motion_feedback": {"step": 7, "accepted": False,
        "target": [4., 8.], "collision_type": "dynamic_0"},
        "position": [4.4, 4.8]}})
    waypoint = controller.select(43, [4.4, 4.8], [10., 8.], graph(), [4., 8.], [4., 2.])
    np.testing.assert_array_equal(waypoint, [4., 2.])

def test_new_anchor_invalidates_old_recovery():
    controller = blocked_controller()
    controller.select(43, [4.4, 4.8], [10., 8.], graph(), [6., 5.], [8., 5.])
    waypoint = controller.select(43, [4.4, 4.8], [4., 0.], graph(), [4., 2.], [4., 2.])
    np.testing.assert_array_equal(waypoint, [4., 2.])

def test_nonzero_origin_uses_physical_floor_lookup():
    frame = GeometryFrame(-3., -7., .4, 12, 12)
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[5, 5] = 1
    controller = recovery(grid, frame)
    assert not controller.segment_safe([-1.2, -5.1], [-.7, -5.1])
    assert controller.segment_safe([-2., -6.], [-1.6, -6.])

from integrations.gppo.scheduler import GPPOTaskScheduler
from utils.policy_adapter import MARVELPolicyAdapter

def test_extended_scheduler_search_action_uses_real_pose_gate():
    scheduler = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler._search_recovery = blocked_controller()
    scheduler.runtime = SimpleNamespace(obstacles=SimpleNamespace(
        frame=GeometryFrame.extended(width_cells=12, height_cells=12)))
    robot = SimpleNamespace(robot_id=43, position=np.array([4.4, 4.8]), heading=0.)
    action = scheduler._search_action(robot, [10., 8.], graph(),
        (np.array([6., 5.]), 1), (np.array([8., 5.]), 0.))
    assert scheduler._search_recovery.segment_safe(robot.position, action[0])
    assert np.linalg.norm(action[0]-robot.position)>.1
    assert not np.array_equal(action[0], [6., 5.])
    delta=action[0]-robot.position
    assert np.isclose(action[1],np.degrees(np.arctan2(delta[1],delta[0]))%360.)

def test_native_scheduler_fallback_action_is_unchanged():
    scheduler = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler._search_recovery = None
    base = (np.array([8., 5.]), 120.)
    robot = SimpleNamespace(robot_id=0, position=np.array([4., 4.]), heading=0.)
    assert scheduler._search_action(robot, [10., 8.], None, None, base) is base

def test_adapter_passes_observations_before_scheduling():
    adapter = MARVELPolicyAdapter.__new__(MARVELPolicyAdapter)
    adapter._using_policy = False
    adapter.agents = []
    adapter.verbose = False
    actions = [(np.array([3., 3.]), 0.)]
    adapter.runtime = SimpleNamespace(default_actions=lambda: actions)
    calls = []
    adapter.scheduler = SimpleNamespace(
        observe=lambda obs: calls.append(("observe", obs)),
        apply=lambda acts: calls.append(("apply", acts)) or acts)
    observations = {0: {"visible_occupancy": [[3, 3, 0]]}}
    assert adapter.get_actions(observations) is actions
    assert calls == [("observe", observations), ("apply", actions)]

from utils.scenario_config import load_and_validate_scenario

def test_runtime_rejection_feedback_and_reset_are_real():
    config = load_and_validate_scenario(ROOT / "configs/scenarios/baseline_exploration.yaml")
    config["environment"]["circular_obstacles"] = []
    config["robots"] = [{"id_range":[0,0], "type":"explorer",
        "config":{"initial_positions":[[4.75,5.5]],"velocity":1.,
                  "fov":120,"sensor_range":10,"yaw_rate":35}}]
    runtime = SimulationRuntime(config)
    runtime.reset()
    grid = np.zeros(runtime.obstacles.frame.shape, dtype=np.uint8)
    grid[5,5] = 1
    runtime.obstacles._file_grid = grid
    runtime.robots[0].position = np.array([4.75,5.5])
    observations, info = runtime.step([(np.array([7.,5.5]),0.)])
    feedback = observations[0]["motion_feedback"]
    assert feedback["accepted"] is False
    assert feedback["collision_type"] == "grid_obstacle"
    np.testing.assert_allclose(feedback["proposal"], [4.85,5.5])
    np.testing.assert_allclose(observations[0]["position"], [4.75,5.5])
    assert len(info["collisions"]) == 1
    assert runtime.reset()[0]["motion_feedback"] == {}

def test_wrong_lattice_labels_cannot_certify_prefix():
    controller = recovery()
    controller.observe({0:{"visible_occupancy":[[2,2,0],[3,2,0]],
        "occupancy_frame":{"origin":[0,0],"cell_size":.4,"width_cells":12,"height_cells":12}}})
    assert not controller.segment_safe([2.,2.],[3.,2.])

def test_only_exact_accepted_pose_infers_current_five_probe_cells():
    controller = recovery()
    controller.observe({0:{"position":[2.5,2.5], "motion_feedback":{
        "step":0,"accepted":True,"grid_probes_clear":True,"proposal":[2.5,2.5],"target":[2.5,2.5]}}})
    assert controller.segment_safe([2.5,2.5],[2.6,2.5])
    assert not controller.segment_safe([2.5,2.5],[3.5,2.5])
    assert set(controller.cells) == {(2,2)}

def test_rejected_or_mismatched_pose_cannot_infer_free_cells():
    for accepted, proposal in ((False,[2.5,2.5]),(True,[4.,4.])):
        controller = recovery()
        controller.observe({0:{"position":[2.5,2.5], "motion_feedback":{
            "step":0,"accepted":accepted,"proposal":proposal,"target":[4.,4.]}}})
        assert not controller.cells

def test_shield_veto_feedback_retains_requested_target():
    runtime, _ = observation_runtime()
    config = load_and_validate_scenario(ROOT / "configs/scenarios/baseline_exploration.yaml")
    config["environment"]["circular_obstacles"] = [{"position":[7.,5.5],"radius":.3}]
    config["robots"] = [{"id_range":[0,0],"type":"explorer",
        "config":{"initial_positions":[[4.75,5.5]],"velocity":1.,
        "fov":120,"sensor_range":10,"yaw_rate":35}}]
    runtime = SimulationRuntime(config)
    runtime.reset()
    observations, _ = runtime.step([(np.array([7.,5.5]),0.)])
    feedback = observations[0]["motion_feedback"]
    assert feedback["shield_rejected"] is True
    assert feedback["requested_target"] == [7.,5.5]
    assert feedback["accepted"] is True  # actual hover passes physical check

def test_shield_veto_is_excluded_even_when_hover_accepted():
    controller = recovery(np.zeros((12,12),dtype=np.uint8))
    controller.select(0,[2.5,2.5],[9.,2.5],None,[7.,2.5],[3.,2.5])
    controller.observe({0:{"position":[2.5,2.5],"motion_feedback":{
        "step":0,"accepted":True,"proposal":[2.5,2.5],"target":[2.5,2.5],
        "requested_target":[7.,2.5],"shield_rejected":True}}})
    result = controller.select(0,[2.5,2.5],[9.,2.5],None,[7.,2.5],[3.,2.5])
    np.testing.assert_array_equal(result,[3.,2.5])

def test_preemption_clears_same_anchor_failed_command():
    controller = recovery(np.zeros((12,12),dtype=np.uint8))
    scheduler = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler._search_recovery = controller
    scheduler.search_assignments = {1:0}
    scheduler.relay_assignments = {}
    controller.select(0,[2.5,2.5],[9.,2.5],None,[7.,2.5],[3.,2.5])
    controller.observe({0:{"position":[2.5,2.5],"motion_feedback":{
        "step":0,"accepted":False,"proposal":[2.6,2.5],"target":[7.,2.5]}}})
    assert scheduler.preempt_uav(0) == ("target_search",1)
    result = controller.select(0,[2.5,2.5],[9.,2.5],None,[7.,2.5],[3.,2.5])
    np.testing.assert_array_equal(result,[7.,2.5])

def test_fresh_search_scenario_clears_observed_recovery_history():
    scheduler = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler._search_recovery = recovery(np.zeros((12,12),dtype=np.uint8))
    scheduler.search_assignments = {1:0}
    scheduler.serviced_heat_ids = {2}
    scheduler.install_search_scenario([])
    assert not scheduler._search_recovery.cells
    assert not scheduler._search_recovery.last_feedback_step

def test_clipped_existing_neighbor_prefix_can_start_without_unseen_cells():
    frame = GeometryFrame.extended(width_cells=500,height_cells=500)
    case = {"position":[95.24917216103731,106.2261870103736],
            "anchor":[28.,474.], "preferred":[88.,112.]}
    controller = ObservedSearchRecovery(frame)
    controller.cells = {(95,106):0, (96,106):0}
    agent = SimpleNamespace(node_coords=np.array([[96.,112.]]),neighbor_indices=np.array([0]))
    target = controller.select(0,case["position"],case["anchor"],agent,
        case["preferred"],[96.,112.])
    length = np.linalg.norm(target-case["position"])
    assert .1 < length <= 1.
    assert controller.segment_safe(case["position"],target)
    assert np.linalg.norm(target-case["anchor"]) < np.linalg.norm(
        np.array(case["position"])-case["anchor"])

def test_accepted_circle_collision_test_does_not_prove_grid_cells_free():
    controller = recovery()
    controller.observe({0:{"position":[2.5,2.5],"motion_feedback":{
        "step":0,"accepted":True,"proposal":[2.5,2.5],"target":[2.5,2.5],
        "grid_probes_clear":False}}})
    assert not controller.cells

from utils.dynamics_models import KinematicSimple
from utils.obstacle_manager import ObstacleManager

def test_sensing_prefix_uses_supported_motion_to_expand_fov_and_continue():
    frame = GeometryFrame.extended(width_cells=500,height_cells=500)
    grid = np.zeros(frame.shape,dtype=np.uint8)
    obstacles = ObstacleManager({"map_size":[500,500]})
    obstacles._file_grid = grid
    controller = ObservedSearchRecovery(frame)
    controller.cells = {(95,106):0,(96,106):0}
    scheduler = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler._search_recovery = controller
    scheduler.runtime = SimpleNamespace(obstacles=obstacles)
    robot = SimpleNamespace(robot_id=0,position=np.array([95.24917216103731,106.2261870103736]),
        heading=350.,velocity=1.)
    initial = robot.position.copy()
    anchor = np.array([28.,474.])
    agent = SimpleNamespace(node_coords=np.array(
        [[92.,104.],[92.,108.],[96.,112.],[96.,104.],[96.,108.]]),
        neighbor_indices=np.arange(5))
    sensor = IdealSensor({"fov":120,"range":10,"los_occlusion":True},frame)
    dynamics = KinematicSimple({"velocity":1.,"yaw_rate":35.})
    moved = []
    for mission in range(4):
        before = robot.position.copy()
        action = scheduler._search_action(robot,anchor,agent,
            (np.array([88.,112.]),0),(np.array([96.,112.]),350.))
        for tick in range(10):
            state,_ = dynamics.step(robot.position,action[0],robot.heading,action[1],1.,.1)
            assert not obstacles.check_collision(state["position"],radius=.2)[0]
            robot.position=state["position"];robot.heading=state["heading"]
            obs=sensor.sense(robot,grid,[(0,robot.position)])
            obs["visible_occupancy"]=[[int(c),int(r),int(grid[r,c])] for c,r in obs["visible_cells"]]
            obs["occupancy_frame"]=frame.describe()
            obs["motion_feedback"]={"step":mission*10+tick,"accepted":True,"grid_probes_clear":True,
                "proposal":robot.position.tolist(),"target":np.asarray(action[0]).tolist()}
            controller.observe({0:obs})
        moved.append(np.linalg.norm(robot.position-before))
    assert all(length>.1 for length in moved)
    assert robot.position[1] > initial[1]+1.
    assert np.linalg.norm(robot.position-anchor)<np.linalg.norm(initial-anchor)

def test_complete_graph_hop_persists_until_actual_pose_reaches_it():
    controller = recovery(np.zeros((12,12),dtype=np.uint8))
    first = controller.select(0,[2.5,2.5],[10.,2.5],None,[7.,2.5],[3.,2.5])
    second = controller.select(0,[3.,2.5],[10.,2.5],None,[2.,3.],[3.,3.])
    np.testing.assert_array_equal(first,[7.,2.5])
    np.testing.assert_array_equal(second,[7.,2.5])

def test_extended_certified_exploration_fallback_preserves_policy_heading():
    controller = recovery(np.zeros((12,12),dtype=np.uint8))
    scheduler = GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    scheduler._search_recovery = controller
    scheduler.runtime = SimpleNamespace(obstacles=SimpleNamespace(frame=controller.frame))
    robot = SimpleNamespace(robot_id=0,position=np.array([2.5,2.5]),heading=0.)
    first = scheduler._search_action(robot,[10.,2.5],None,None,(np.array([3.,2.5]),150.))
    assert first[1] == 150.
    robot.position = np.array([2.6,2.5])
    second = scheduler._search_action(robot,[10.,2.5],None,(np.array([4.,3.]),1),
        (np.array([3.,3.]),40.))
    np.testing.assert_array_equal(second[0],[3.,2.5])
    assert second[1] == 150.

from utils.dynamics_models import KinematicWithAccel

def test_extended_recovery_rejects_unsupported_acceleration_dynamics():
    runtime=SimpleNamespace(obstacles=SimpleNamespace(
        frame=GeometryFrame.extended(width_cells=12,height_cells=12)),
        dynamics=KinematicWithAccel(),dynamics_by_type={})
    with __import__("pytest").raises(ValueError,match="KinematicSimple"):
        GPPOTaskScheduler._make_search_recovery(runtime)

def test_native_recovery_bypasses_model_guard():
    runtime=SimpleNamespace(obstacles=SimpleNamespace(
        frame=GeometryFrame(0,0,.4,12,12,mode="marvel_native")),
        dynamics=KinematicWithAccel(),dynamics_by_type={})
    assert GPPOTaskScheduler._make_search_recovery(runtime) is None

def test_ordinary_extended_runtime_keeps_explicit_legacy_radius_tasks():
    runtime,_=observation_runtime()
    runtime.config={}
    runtime.tasks=SimpleNamespace(tasks={})
    assert runtime._get_target_detections({}) is None

def test_native_target_detection_stays_assignment_gated():
    runtime,_=observation_runtime("marvel_native")
    runtime.config={}
    runtime.active_search_uav_ids=set()
    runtime.tasks=SimpleNamespace(tasks={"search":SimpleNamespace(
        task_type="target_search",task_id="search")})
    runtime.target_detector=SimpleNamespace(detect=lambda task,obs,ids:{0:0} if 0 in ids else {})
    assert runtime._get_target_detections({0:{"visible_cells":[[3,5]]}})=={"search":{}}

def test_recovery_avoids_only_observed_nearby_uav_positions():
    controller=recovery(np.zeros((12,12),dtype=np.uint8))
    controller.observe({0:{'position':[2.5,2.5],'nearby_robot_positions':{1:[6.,2.5]}}})
    agent=SimpleNamespace(node_coords=np.array([[8.,2.5],[6.,6.5]]),neighbor_indices=np.array([0,1]))
    target=controller.select(0,[2.5,2.5],[10.,2.5],agent,[8.,2.5],[8.,2.5])
    assert not np.array_equal(target,[8.,2.5])
    assert target[1]>2.5

def test_runtime_publishes_only_existing_range_observed_peer_positions():
    runtime,_=observation_runtime()
    runtime.robots.extend([RobotState(1,'uav',np.array([4.,5.]),1.,0.),RobotState(9,'uav',np.array([20.,20.]),1.,0.)])
    obs=runtime._get_observations()[0]
    assert obs['nearby_robot_positions']=={1:[4.,5.]}

def test_assignment_reset_preserves_current_nearby_measurements():
    controller=recovery(np.zeros((12,12),dtype=np.uint8))
    controller.observe({0:{'position':[2.5,2.5],'nearby_robot_positions':{1:[6.,2.5]}}})
    controller.sync_assignments({7:0})
    target=controller.select(0,[2.5,2.5],[10.,2.5],None,[8.,2.5],[8.,2.5])
    assert not np.array_equal(target,[8.,2.5])


def test_occupancy_areal_semantics_are_explicit_and_reset_with_episode():
    runtime,_=observation_runtime();runtime.obstacles.has_occupancy_map=True
    obs=runtime._get_observations()
    controller=ObservedSearchRecovery(runtime.obstacles.frame);controller.observe(obs)
    assert controller._authoritative_areal_grid is True
    runtime.obstacles.has_occupancy_map=False;controller.observe(runtime._get_observations())
    assert controller._authoritative_areal_grid is False
    controller._authoritative_areal_grid=True;controller.reset()
    assert controller._authoritative_areal_grid is False
