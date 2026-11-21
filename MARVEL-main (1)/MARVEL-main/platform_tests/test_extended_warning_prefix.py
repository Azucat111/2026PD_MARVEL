from types import SimpleNamespace as S
import numpy as np
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_travel import extended_observed_travel_time_matrix
from integrations.gppo.task_graph import UAVState,Subtask,TaskType
from utils.geometry import GeometryFrame

def warning_hit(a,b,step=0):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);center=np.array([3.,2.]);d=b-a
    t=float(np.clip(np.dot(center-a,d)/max(np.dot(d,d),1e-20),0,1))
    return np.linalg.norm(a+t*d-center)<=.8

def recovery():
    r=ObservedSearchRecovery(GeometryFrame(0,0,1,12,12));r.cells={(x,y):0 for x in range(12) for y in range(12)}
    r.anchors[0]=np.array([6.,2.]);r.prefixes[0]=np.array([4.,2.])
    return r

def test_search_and_relay_reject_cached_warning_crossing_before_prefix_choice():
    r=recovery();s=GPPOTaskScheduler.__new__(GPPOTaskScheduler);s._search_recovery=r
    s.runtime=S(current_step=20,obstacles=S(frame=r.frame));s.safety=S(hazards=S(segment_intersects_warning_buffer=warning_hit))
    s.search_reach_radius=3.5
    robot=S(robot_id=0,position=np.array([2.,2.]),heading=0.)
    agent=S(node_coords=np.array([[3.,4.]]),neighbor_indices=np.array([0]))
    action=s._search_action(robot,np.array([6.,2.]),agent,(np.array([4.,2.]),0),(robot.position.copy(),0.))
    np.testing.assert_allclose(action[0],[3.,4.])
    assert not warning_hit(robot.position,action[0])
    assert r.segment_safe(robot.position,[4.,2.])  # Physical FREE proof remains usable by Safety escape.

def test_public_warning_edges_change_extended_travel_without_truth_map():
    r=recovery();coords=np.array([[2.,2.],[6.,2.],[2.,5.],[6.,5.]])
    adj=np.ones((4,4));np.fill_diagonal(adj,0)
    for a,b in [(0,1),(0,2),(2,3),(3,1)]:adj[a,b]=adj[b,a]=0
    agent=S(node_coords=coords,adjacent_matrix=adj,current_index=0,neighbor_indices=np.array([0,1,2]))
    u=[UAVState(0,position=(2.,2.),velocity=1.)];tasks=[Subtask(100,TaskType.TARGET_SEARCH,5,position=(6.,2.))]
    ordinary=extended_observed_travel_time_matrix(u,tasks,{0:agent},prefix_controller=r)
    guarded=extended_observed_travel_time_matrix(u,tasks,{0:agent},prefix_controller=r,segment_filter=lambda a,b:not warning_hit(a,b))
    assert ordinary[0,0]==4. and guarded[0,0]==10.


def test_ordinary_prefix_also_rechecks_current_public_warning():
    from platform_tests.test_extended_final_motion_authority import make_adapter
    adapter,r=make_adapter(token='exploration')
    adapter.runtime.current_step=20
    adapter.scheduler.safety.hazards=S(segment_intersects_warning_buffer=warning_hit)
    adapter.agents=[S(node_coords=np.array([[3.,4.]]),neighbor_indices=np.array([0]))]
    result=adapter._certify_extended_actions([(np.array([4.,2.]),90.)])
    np.testing.assert_allclose(result[0][0],[3.,4.])
