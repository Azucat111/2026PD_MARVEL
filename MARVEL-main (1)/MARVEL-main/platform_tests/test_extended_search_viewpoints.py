"""Observed-only peripheral sensing; movement must retain existing certification."""
from types import SimpleNamespace
import math
import numpy as np
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.search_viewpoints import ObservedSearchViewpoints

def fixture():
 frame=GeometryFrame.extended(width_cells=40,height_cells=40)
 rec=ObservedSearchRecovery(frame);rec._authoritative_areal_grid=True
 rec.cells={(x,y):0 for y in range(40) for x in range(40)}
 rec.cells.pop((23,16)) # A potential target cell is still UNKNOWN, never truth input.
 rec.observed_positions[28]=np.array([16.,8.]);rec.nearby_positions[28]=[]
 agent=SimpleNamespace(node_coords=np.array([[16.,8.],[20.,8.]]),adjacent_matrix=np.zeros((2,2)),current_index=0,neighbor_indices=np.array([0,1]))
 heat=np.array([22.,18.]);views=ObservedSearchViewpoints(frame,sensor_range=10.,fov=120.,prior_radius=5.5)
 # Composite historical Search scan memory; exact target is never specified.
 checked=[(x,y,0) for (x,y),v in rec.cells.items() if np.linalg.norm(frame.cell_to_world((x,y))-heat)<=5.5]
 views.observe({28:dict(position=[16,8],heading=0,visible_occupancy=checked,occupancy_frame=frame.describe())},{2:28})
 return frame,rec,agent,heat,views

def test_unknown_prior_can_add_sensing_value_without_becoming_a_motion_certificate():
 _,rec,agent,heat,views=fixture()
 result=views.select(28,2,heat,agent,rec)
 assert result is not None
 np.testing.assert_allclose(result.position,[20,8])
 assert rec.segment_safe(rec.observed_positions[28],result.position)
 assert (23,16) not in rec.cells
 assert math.isclose(result.heading,78.69006752597979)

def test_unknown_cell_on_motion_path_rejects_the_same_sensing_candidate():
 _,rec,agent,heat,views=fixture();rec.cells.pop((18,8))
 assert views.select(28,2,heat,agent,rec) is None

def test_public_warning_filter_remains_authoritative():
 _,rec,agent,heat,views=fixture()
 assert views.select(28,2,heat,agent,rec,segment_filter=lambda a,b:False) is None

def test_missing_current_self_observation_cannot_issue_a_route():
 _,rec,agent,heat,views=fixture();rec.observed_positions.clear()
 assert views.select(28,2,heat,agent,rec) is None

def test_nonsearch_shared_labels_do_not_count_as_search_scan_memory():
 frame,rec,agent,heat,views=fixture()
 views.observe({7:dict(position=[20,8],heading=0,visible_occupancy=[[23,16,0]],occupancy_frame=frame.describe())},{2:28})
 assert (23,16) not in views.checked_cells_by_heat[2]
 assert views.select(28,2,heat,agent,rec) is not None

def test_actual_search_observation_removes_that_cell_from_remaining_sensing_utility():
 frame,rec,agent,heat,views=fixture()
 views.observe({28:dict(position=[20,8],heading=0,visible_occupancy=[[23,16,0]],occupancy_frame=frame.describe())},{2:28})
 assert (23,16) in views.checked_cells_by_heat[2]
 assert views.select(28,2,heat,agent,rec) is None

def test_heat_scan_memory_survives_owner_change_and_remains_heat_scoped():
 frame,rec,agent,heat,views=fixture()
 views.observe({9:dict(position=[20,8],heading=0,visible_occupancy=[[23,16,0]],occupancy_frame=frame.describe())},{2:9})
 assert (23,16) in views.checked_cells_by_heat[2]
 assert (23,16) not in views.checked_cells_by_heat.get(3,set())

def test_invalid_observation_frame_does_not_pollute_coverage():
 _,rec,agent,heat,views=fixture()
 views.observe({28:dict(position=[20,8],heading=0,visible_occupancy=[[23,16,0]],occupancy_frame={'origin':[100,100]})},{2:28})
 assert (23,16) not in views.checked_cells_by_heat[2]

def scheduler_fixture():
 from integrations.gppo.scheduler import GPPOTaskScheduler
 frame,rec,agent,heat,views=fixture()
 sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
 sched._search_recovery=rec;sched._search_viewpoints=views;sched._peripheral_search_targets={}
 sched.search_reach_radius=3.5;sched.search_assignments={2:28}
 sched.runtime=SimpleNamespace(current_step=400,geometry_mode='extended',obstacles=SimpleNamespace(frame=frame))
 sched.safety=SimpleNamespace(hazards=SimpleNamespace(warning_margin=5.,risk_at=lambda p,step:SimpleNamespace(clearance=-1.),segment_intersects_warning_buffer=lambda a,b,step:False))
 robot=SimpleNamespace(robot_id=28,position=np.array([16.,8.]),heading=0.)
 return sched,robot,agent,heat

def test_excluded_service_disk_routes_to_observed_viewpoint_but_camera_keeps_public_heat():
 sched,robot,agent,heat=scheduler_fixture()
 goal,heading=sched._search_action(robot,heat,agent,None,(np.array([16.,12.]),0.),route_goal_radius=3.5)
 np.testing.assert_allclose(goal,[20,8])
 assert math.isclose(heading,59.03624346792648)
 assert sched.search_assignments=={2:28}

def test_warning_that_does_not_exclude_whole_service_disk_keeps_original_control_scope():
 sched,robot,agent,heat=scheduler_fixture()
 sched.safety.hazards.risk_at=lambda p,step:SimpleNamespace(clearance=2.)
 assert sched._peripheral_search_target(robot,heat,agent) is None
 assert not sched._peripheral_search_targets

def test_released_search_owner_cannot_continue_cached_peripheral_motion_as_relay():
 sched,robot,agent,heat=scheduler_fixture()
 assert sched._peripheral_search_target(robot,heat,agent) is not None
 sched.search_assignments={}
 assert sched._peripheral_search_target(robot,heat,agent) is None
 assert 28 not in sched._peripheral_search_targets

def test_native_action_bypasses_all_peripheral_viewpoint_selection():
 sched,robot,agent,heat=scheduler_fixture();sched._search_recovery=None;sched.runtime.geometry_mode='marvel_native'
 def forbidden(*args):raise AssertionError('Native invoked extended peripheral selection')
 sched._peripheral_search_target=forbidden
 goal,_=sched._search_action(robot,heat,agent,(np.array([18.,8.]),1),(np.array([16.,12.]),0.))
 np.testing.assert_allclose(goal,[18,8])
