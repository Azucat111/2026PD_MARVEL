import numpy as np
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
from integrations.gppo.search_deadline import apply_search_deadline_bound

def graph():
    return TaskGraph([UAVState(0,position=(373.87,0),velocity=1),UAVState(1,position=(300,0),velocity=1)],
        [Subtask(0,TaskType.TARGET_SEARCH,5,position=(0,0)),Subtask(1,TaskType.RELAY,2,position=(0,0))])

def test_proven_impossible_only_and_relay_untouched():
    g=graph();reasons=apply_search_deadline_bound(g,300,10,5.5,1,0.05)
    assert g.action_mask[0].tolist()==[True,False]
    assert not g.action_mask[1].any()
    assert reasons[0]['uav_id']==0
    assert reasons[0]['lower_bound_seconds']>300

def test_stale_displacement_and_grid_cell_allowance_are_conservative():
    g=TaskGraph([UAVState(0,position=(12.5,0),velocity=1)],
        [Subtask(0,TaskType.TARGET_SEARCH,5,position=(0,0))])
    apply_search_deadline_bound(g,0,10,0,1,2)
    assert not g.action_mask.any() # Potentially already within sensor cells after stale displacement.

def test_existing_masks_remain_and_unknown_goal_not_declared_impossible():
    g=graph();g.action_mask[0,1]=True;g.subtasks[0].position=None
    assert apply_search_deadline_bound(g,0,10,5.5,1,0)==[]
    assert g.action_mask[0].tolist()==[False,True]

def test_boundary_is_eligible_and_remaining_horizon_is_respected():
    g=TaskGraph([UAVState(0,position=(16,0),velocity=2)],
        [Subtask(0,TaskType.TARGET_SEARCH,5,position=(0,0))])
    apply_search_deadline_bound(g,3,10,0,0,0)
    assert not g.action_mask.any()
    apply_search_deadline_bound(g,2.9,10,0,0,0)
    assert g.action_mask.all()

from types import SimpleNamespace
from integrations.gppo.event_allocator import GPPOEventAllocator
from utils.dynamics_models import KinematicSimple,KinematicWithAccel
from utils.sensor_models import IdealSensor

def allocator(mode):
    a=GPPOEventAllocator.__new__(GPPOEventAllocator)
    a.runtime=SimpleNamespace(geometry_mode=mode,max_steps=3000,current_step=0,dt=.1,
        robots=[SimpleNamespace(robot_id=i,robot_type='explorer') for i in (0,1)],
        dynamics=KinematicSimple({'velocity':1}),dynamics_by_type={},
        sensor_by_type={'explorer':IdealSensor({'range':10})},
        config={'_gppo_profile':{'search_sensor_range':10,'search_guaranteed_hidden_radius':5.5,'search_cell_size':1}})
    return a

def test_allocator_native_and_missing_age_are_unchanged():
    g=graph();allocator('marvel_native')._apply_deadline_mask(g,.05)
    assert not g.action_mask.any()
    allocator('extended')._apply_deadline_mask(g,None)
    assert not g.action_mask.any()

def test_allocator_extended_uses_physics_remaining_horizon():
    g=graph();a=allocator('extended');a.runtime.current_step=1000
    a._apply_deadline_mask(g,.05)
    assert g.action_mask[0].all()
    assert not g.action_mask[1].any()
    assert a.last_deadline_masks[0]['remaining_seconds']==200


def test_accelerating_model_has_no_certified_constant_speed_bound():
    a=allocator('extended');a.runtime.dynamics=KinematicWithAccel({});g=graph()
    a._apply_deadline_mask(g,0)
    assert not g.action_mask.any()

def test_actual_larger_sensor_range_prevents_false_time_infeasibility():
    a=allocator('extended');a.runtime.current_step=2990
    a.runtime.sensor_by_type={'explorer':IdealSensor({'range':20})}
    g=TaskGraph([UAVState(0,position=(20,0),velocity=1)],
        [Subtask(0,TaskType.TARGET_SEARCH,5,position=(0,0))])
    a._apply_deadline_mask(g,0)
    assert not g.action_mask.any()
