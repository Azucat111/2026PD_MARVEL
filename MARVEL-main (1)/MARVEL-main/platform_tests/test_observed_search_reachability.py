import numpy as np
from utils.geometry import GeometryFrame
from integrations.gppo.observed_reachability import ObservedReachabilityGate

def setup():
    frame=GeometryFrame(-2,-3,1,20,10)
    cells={(9,row):1 for row in range(10)}
    return frame,cells

def test_closed_observed_wall_proves_detection_region_unreachable():
    f,cells=setup();gate=ObservedReachabilityGate(f)
    result=gate.reason(cells,(1,2),(15,2),1,0,1,1)
    assert result is not None
    assert result['minimum_heat_distance_m']>result['optimistic_detection_allowance_m']

def test_unknown_door_must_keep_pair_feasible():
    f,cells=setup();cells.pop((9,4));gate=ObservedReachabilityGate(f)
    assert gate.reason(cells,(1,2),(15,2),1,0,1,1) is None

def test_sensor_support_across_wall_is_not_declared_impossible():
    f,cells=setup();gate=ObservedReachabilityGate(f)
    assert gate.reason(cells,(1,2),(8,2),1,0,10,1) is None

def test_stale_position_uncertainty_union_and_nonfinite_fail_open():
    f,cells=setup();gate=ObservedReachabilityGate(f)
    assert gate.reason(cells,(6.9,2),(15,2),1,1.2,1,1) is None
    assert gate.reason(cells,(np.nan,2),(15,2),1,0,1,1) is None

def test_observation_change_rebuilds_components():
    f,cells=setup();gate=ObservedReachabilityGate(f)
    assert gate.reason(cells,(1,2),(15,2),1,0,1,1)
    cells[(9,4)]=0
    assert gate.reason(cells,(1,2),(15,2),1,0,1,1) is None

from types import SimpleNamespace
from integrations.gppo.event_allocator import GPPOEventAllocator
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
from integrations.gppo.scheduler import GPPOTaskScheduler
from utils.dynamics_models import KinematicSimple,KinematicWithAccel
from utils.sensor_models import IdealSensor

def runtime(mode='extended'):
    f,cells=setup();controller=SimpleNamespace(frame=f,cells=cells,_authoritative_areal_grid=True)
    return SimpleNamespace(geometry_mode=mode,current_step=0,dt=.1,
        robots=[SimpleNamespace(robot_id=7,robot_type='explorer',velocity=1)],
        dynamics=KinematicSimple({'velocity':1}),dynamics_by_type={},
        sensor_by_type={'explorer':IdealSensor({'range':1})},
        config={'_gppo_profile':{'search_sensor_range':1,'search_guaranteed_hidden_radius':1}},
        high_level_scheduler=SimpleNamespace(_search_recovery=controller))

def test_allocator_masks_only_proven_search_pairs_and_preserves_native():
    g=TaskGraph([UAVState(7,position=(1,2))],
        [Subtask(1,TaskType.TARGET_SEARCH,5,position=(15,2)),Subtask(2,TaskType.RELAY,2,position=(15,2))])
    a=GPPOEventAllocator.__new__(GPPOEventAllocator);a.runtime=runtime('marvel_native')
    a._apply_observed_reachability_mask(g,0)
    assert not g.action_mask.any()
    a.runtime=runtime();a._apply_observed_reachability_mask(g,0)
    assert g.action_mask[:,0].tolist()==[True,False]

def test_scheduler_releases_unreachable_assignment_without_servicing_heat():
    s=GPPOTaskScheduler.__new__(GPPOTaskScheduler);s.runtime=runtime();s.runtime.high_level_scheduler=s
    s._search_recovery=SimpleNamespace(frame=setup()[0],cells=setup()[1],_authoritative_areal_grid=True)
    s.profile=SimpleNamespace(stale_age_fraction=.05,high_level_interval_steps=10)
    s._last_observed_step=0;s.runtime.robots=[SimpleNamespace(robot_id=7,robot_type='explorer',velocity=1)]
    s.search_assignments={1:7};s.heat_by_id={1:SimpleNamespace(position=(15,2))}
    s.search_dwell_by_heat={1:3};s.serviced_heat_ids=set()
    released=s._release_proven_unreachable_search({7:np.array([1,2])})
    assert released==[(1,7)]
    assert s.search_assignments=={} and s.serviced_heat_ids==set()
    assert s.search_dwell_by_heat[1]==0

from integrations.gppo.observed_reachability import observed_search_impossibility

def test_endpoint_model_can_jump_cells_at_high_speed_so_gate_fails_open():
    rt=runtime();rt.dynamics=KinematicSimple({'velocity':25})
    a=GPPOEventAllocator.__new__(GPPOEventAllocator);a.runtime=rt
    g=TaskGraph([UAVState(7,position=(1,2))],[Subtask(1,TaskType.TARGET_SEARCH,5,position=(15,2))])
    a._apply_observed_reachability_mask(g,0)
    assert not g.action_mask.any()


def test_non_areal_circle_samples_must_not_prove_a_closed_cut():
    rt=runtime();rt.high_level_scheduler._search_recovery._authoritative_areal_grid=False
    a=GPPOEventAllocator.__new__(GPPOEventAllocator);a.runtime=rt
    g=TaskGraph([UAVState(7,position=(1,2))],[Subtask(1,TaskType.TARGET_SEARCH,5,position=(15,2))])
    a._apply_observed_reachability_mask(g,0)
    assert not g.action_mask.any()

def test_four_connected_mode_cannot_invent_a_stale_start_across_a_corner():
    f=GeometryFrame(0,0,1,20,10);cells={(9,row):1 for row in range(10)}
    cells.pop((9,4));cells[(8,4)]=1;cells[(9,5)]=1
    gate=ObservedReachabilityGate(f);gate.prepare(cells,connectivity=4)
    assert gate.reason(cells,(8.5,5.5),(18,5.5),1,.05,1,1,prepared=True)


def test_motion_goal_closed_cut_has_no_sensor_detection_allowance():
    from integrations.gppo.observed_reachability import observed_motion_impossibility
    r=runtime();result=observed_motion_impossibility(r,7,(1,2),(15,2),0)
    assert result is not None and result['reason']=='observed_closed_cut_outside_motion_goal_support'
    r.high_level_scheduler._search_recovery.cells.pop((9,4))
    r.current_step+=1
    assert observed_motion_impossibility(r,7,(1,2),(15,2),0) is None

def test_native_motion_mask_preserves_frozen_and_unknown_map_fails_open():
    from integrations.gppo.observed_reachability import observed_motion_impossibility
    assert observed_motion_impossibility(runtime('marvel_native'),7,(1,2),(15,2),0) is None
    r=runtime();r.high_level_scheduler._search_recovery._authoritative_areal_grid=False
    assert observed_motion_impossibility(r,7,(1,2),(15,2),0) is None
