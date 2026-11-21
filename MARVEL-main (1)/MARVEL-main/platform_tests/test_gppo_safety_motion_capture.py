from types import SimpleNamespace as NS
import json
import numpy as np
from scripts.gppo_motion_capture import capture_motion_frame,json_value,append_motion_frame

class GuardedRuntime:
    current_step=12
    dt=.1
    robots=[NS(robot_id=7,position=np.array([1.,2.]),heading=.5,travel_distance=3.)]
    _motion_feedback={7:{'moved_m':.2}}
    def __getattr__(self,name):
        raise AssertionError('Unexpected runtime access: '+name)

def fixture():
    recovery=NS(frame=NS(describe=lambda:{'cell_size':.4}),cells={(2,3):0},
        prefixes={7:[np.array([2.,2.])]},prefix_headings={7:.2},anchors={7:np.array([3.,3.])},failed={})
    return NS(search_assignments={100:7},relay_assignments={},
        heat_by_id={100:NS(position=np.array([3.,3.]))},_search_recovery=recovery)

def test_safety_only_uav_retains_own_observation_and_observed_graph():
    scheduler=fixture();scheduler.search_assignments={};scheduler.safety=NS(active_uav_ids={7});scheduler._marvel_agents={7:NS(node_coords=np.array([[1.,2.],[2.,2.]]),neighbor_indices=np.array([1]),current_index=0,adjacent_matrix=np.zeros((2,2)))}
    frame=capture_motion_frame(GuardedRuntime(),scheduler,{7:{'heading':.5,'position':np.array([1.,2.])},8:{'heading':1.}})
    assert set(frame['selected_own_observations'])=={7}
    assert set(frame['observed_graphs_diagnostic_only'])=={7}

def test_capture_retains_observed_peripheral_scan_memory_for_reconstruction():
    scheduler=fixture();scheduler._peripheral_search_targets={7:(100,np.array([2.,2.]))};scheduler._search_viewpoints=NS(checked_cells_by_heat={100:{(2,3)}})
    frame=capture_motion_frame(GuardedRuntime(),scheduler,{7:{'heading':.5}})
    state=frame['observed_search_viewpoint_state']
    assert state['checked_cells_by_heat'][100]=={(2,3)}
    np.testing.assert_array_equal(state['cached_targets'][7][1],[2.,2.])
