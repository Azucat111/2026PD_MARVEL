"""Extended travel estimates must follow the observed route, native frozen stays."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.task_graph import UAVState,Subtask,TaskType
from integrations.gppo.observed_graph import observed_travel_time_matrix
from integrations.gppo.extended_travel import extended_observed_travel_time_matrix

def u_route():
    coords=np.array([[2.5,2.5],[6.5,2.5],[6.5,6.5],[2.5,6.5]])
    adj=np.ones((4,4),dtype=int);np.fill_diagonal(adj,0)
    for i,j in ((0,1),(1,2),(2,3)):adj[i,j]=adj[j,i]=0
    return SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=0,neighbor_indices=np.array([0,1]))

def test_frozen_cost_is_preserved_despite_observed_u_detour():
    uavs=[UAVState(0,position=(2.5,2.5),velocity=1.)]
    tasks=[Subtask(7,TaskType.TARGET_SEARCH,5,position=(2.5,6.5))]
    assert observed_travel_time_matrix(uavs,tasks,{0:u_route()})[0,0]==4.

def test_extended_cost_uses_closest_reachable_goal_then_known_path():
    uavs=[UAVState(0,position=(2.5,2.5),velocity=1.)]
    tasks=[Subtask(7,TaskType.TARGET_SEARCH,5,position=(2.5,6.5))]
    cost=extended_observed_travel_time_matrix(uavs,tasks,{0:u_route()})
    assert cost[0,0]==12.

def test_extended_cost_adds_unknown_residual_only_after_observed_destination():
    uavs=[UAVState(0,position=(2.5,2.5),velocity=2.)]
    tasks=[Subtask(7,TaskType.TARGET_SEARCH,5,position=(2.5,7.5))]
    assert extended_observed_travel_time_matrix(uavs,tasks,{0:u_route()})[0,0]==6.5

def test_real_pose_attaches_only_to_supported_visible_neighbor_directions():
    from integrations.gppo.search_recovery import ObservedSearchRecovery
    from utils.geometry import GeometryFrame
    controller=ObservedSearchRecovery(GeometryFrame.extended(width_cells=12,height_cells=12))
    controller.cells={(col,row):int(col==2 and row==4) for col in range(12) for row in range(12)}
    uavs=[UAVState(0,position=(2.8,2.5),velocity=1.)]
    tasks=[Subtask(7,TaskType.TARGET_SEARCH,5,position=(2.5,6.5))]
    diagnostics={}
    cost=extended_observed_travel_time_matrix(uavs,tasks,{0:u_route()},controller,diagnostics)
    np.testing.assert_allclose(cost[0,0],11.7,rtol=1e-6)
    assert diagnostics[7,0]['unknown_residual_lower_bound_m']==0.

def test_missing_visible_attachment_is_only_a_lower_bound_not_no_solution():
    from integrations.gppo.search_recovery import ObservedSearchRecovery
    from utils.geometry import GeometryFrame
    controller=ObservedSearchRecovery(GeometryFrame.extended(width_cells=12,height_cells=12))
    uavs=[UAVState(0,position=(2.8,2.5),velocity=1.)]
    tasks=[Subtask(7,TaskType.TARGET_SEARCH,5,position=(2.5,6.5))]
    diagnostics={}
    cost=extended_observed_travel_time_matrix(uavs,tasks,{0:u_route()},controller,diagnostics)
    np.testing.assert_allclose(cost[0,0],np.hypot(.3,4),rtol=1e-6)
    assert diagnostics[7,0]['status']=='no_certified_attachment_euclidean_lower_bound'

def test_extended_allocator_uses_corrected_cost_and_native_keeps_frozen(monkeypatch):
    import integrations.gppo.event_allocator as module
    from integrations.gppo.event_allocator import GPPOEventAllocator,GPPOEventSlot
    from integrations.gppo.task_graph import TaskGraph
    from integrations.gppo.search_recovery import ObservedSearchRecovery
    from integrations.rule_scheduling import choose_rule_assignment
    from utils.geometry import GeometryFrame
    state=TaskGraph([UAVState(0,position=(2.5,2.5),velocity=1.)],[],current_time=0)
    monkeypatch.setattr(module,'RuntimeGraphBuilder',lambda *args,**kwargs:SimpleNamespace(build=lambda:SimpleNamespace(graph=state)))
    controller=ObservedSearchRecovery(GeometryFrame.extended(width_cells=12,height_cells=12))
    controller.cells={(col,row):int(col==2 and row==4) for col in range(12) for row in range(12)}
    allocator=GPPOEventAllocator.__new__(GPPOEventAllocator)
    allocator._marvel_agents={0:u_route()};allocator.adapter=SimpleNamespace(assign=choose_rule_assignment)
    slot=GPPOEventSlot(7,TaskType.TARGET_SEARCH,5,position=(2.5,6.5),processing_time=6)
    for mode,expected in (('extended',18.),('marvel_native',10.)):
        allocator.runtime=SimpleNamespace(current_step=0,dt=.1,geometry_mode=mode,
            high_level_scheduler=SimpleNamespace(_search_recovery=controller))
        assert allocator.allocate([slot])[0].total_time==expected


def certificate_case(actual):
    from integrations.gppo.search_recovery import ObservedSearchRecovery
    from utils.geometry import GeometryFrame
    controller=ObservedSearchRecovery(GeometryFrame.extended(width_cells=12,height_cells=12))
    controller.cells={(col,row):int((col,row)==(5,5)) for col in range(12) for row in range(12)}
    if actual is not None:controller.observe({0:{'position':actual}})
    agent=SimpleNamespace(node_coords=np.array([[6.,4.75]]),adjacent_matrix=np.zeros((1,1)),
        current_index=0,neighbor_indices=np.array([0]))
    diagnostics={}
    extended_observed_travel_time_matrix([UAVState(0,position=(4.8,4.79),velocity=1.)],
        [Subtask(7,TaskType.TARGET_SEARCH,5,position=(8.,4.75))],{0:agent},controller,diagnostics)
    return controller,diagnostics[7,0]

def test_delayed_clear_pose_cannot_certify_blocked_own_current_pose():
    controller,details=certificate_case([4.8,4.84])
    assert controller.segment_safe([4.8,4.79],[6.,4.75])
    assert not controller.segment_safe([4.8,4.84],[6.,4.75])
    assert details['real_pose_clearance_certified'] is False

def test_missing_own_current_measurement_cannot_certify_real_pose():
    _,details=certificate_case(None)
    assert details['real_pose_clearance_certified'] is False

def test_own_clear_pose_certifies_actual_first_attachment():
    controller,details=certificate_case([4.8,4.79])
    assert details['real_pose_clearance_certified'] is True
    controller.reset_uid(0)
    np.testing.assert_array_equal(controller.observed_positions[0],[4.8,4.79])
    controller.reset()
    assert controller.observed_positions=={}


def test_certificate_tracks_chosen_path_source_not_another_clear_attachment():
    controller,_=certificate_case([4.8,4.84])
    coords=np.array([[4.,4.],[6.,4.75]])
    agent=SimpleNamespace(node_coords=coords,adjacent_matrix=np.eye(2,dtype=int)*0+np.array([[0,1],[1,0]]),
        current_index=0,neighbor_indices=np.array([0,1]))
    diagnostics={}
    extended_observed_travel_time_matrix([UAVState(0,position=(4.8,4.79),velocity=1.)],
        [Subtask(7,TaskType.TARGET_SEARCH,5,position=(8.,4.75))],{0:agent},controller,diagnostics)
    assert controller.segment_safe([4.8,4.84],coords[0])
    assert diagnostics[7,0]['observed_destination']==coords[1].tolist()
    assert diagnostics[7,0]['real_pose_clearance_certified'] is False

def test_omitted_uid_measurement_does_not_retain_current_pose_certificate():
    controller,_=certificate_case([4.8,4.79])
    controller.observe({1:{'position':[1.,1.]}})
    assert 0 not in controller.observed_positions
