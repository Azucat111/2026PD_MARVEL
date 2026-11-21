"""Sensor-only conservative physical-cell to MARVEL-cell conversion."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from parameter import FREE,OCCUPIED,UNKNOWN
from integrations.observed_belief import ObservedBeliefMap

def make(origin=(0.,0.)):
    frame=GeometryFrame(origin[0],origin[1],1.,3,3,mode='extended')
    return frame,ObservedBeliefMap(frame,origin,.4,(8,8))

def observation(frame,labels):
    return {0:{'occupancy_frame':frame.describe(),'visible_occupancy':labels}}

def test_visible_obstacle_is_occupied_and_unknown_neighbor_not_free():
    frame,belief=make()
    belief.observe(observation(frame,[[0,0,0],[1,0,1]]))
    raster=belief.rasterize()
    assert raster[0,0]==FREE
    assert raster[0,2]==OCCUPIED  # [.8,1.2) overlaps occupied [1,2)
    assert raster[0,5]==UNKNOWN  # [2,2.4) unseen
    assert not belief.point_clear([1.1,.4])

def test_free_requires_every_overlapping_physical_cell_observed():
    frame,belief=make()
    belief.observe(observation(frame,[[0,0,0]]))
    assert belief.rasterize()[0,2]==UNKNOWN
    belief.observe(observation(frame,[[1,0,0]]))
    assert belief.rasterize()[0,2]==FREE

def test_nonzero_origin_and_exact_boundary_do_not_bleed():
    frame,belief=make((10.,-3.))
    belief.observe(observation(frame,[[0,0,0],[1,0,1]]))
    raster=belief.rasterize()
    assert raster[0,4]==OCCUPIED
    assert raster[0,5]==UNKNOWN
    assert raster[-1,-1]==UNKNOWN

def test_admission_uses_radius_probes_not_truth_or_center_only():
    frame,belief=make()
    belief.observe(observation(frame,[[0,0,0],[1,0,1]]))
    assert belief.point_clear([.5,.5])
    assert not belief.point_clear([.9,.5])
    assert not belief.point_clear([.5,1.5])

from utils.policy_adapter import MARVELPolicyAdapter
from types import SimpleNamespace

def test_adapter_updates_typed_belief_before_policy_and_scheduler():
    frame,observed=make()
    adapter=MARVELPolicyAdapter.__new__(MARVELPolicyAdapter)
    adapter._observed_belief=observed
    adapter.belief_map=np.full((8,8),UNKNOWN,dtype=np.int32)
    adapter._using_policy=True; adapter.agents=[object()]; adapter.verbose=False
    events=[]
    def policy(obs):
        assert adapter.belief_map[0,2]==OCCUPIED
        events.append('policy'); return []
    adapter._policy_actions=policy
    adapter.scheduler=SimpleNamespace(observe=lambda obs:events.append('observe'),apply=lambda actions:events.append('apply'))
    adapter.get_actions(observation(frame,[[0,0,0],[1,0,1]]))
    assert events[-1]=='apply'


def test_ordinary_relay_prefix_is_certified_from_real_pose():
    frame,observed=make()
    observed.observe(observation(frame,[[0,0,0],[1,0,1]]))
    adapter=MARVELPolicyAdapter.__new__(MARVELPolicyAdapter)
    adapter._observed_belief=observed;adapter._motion_roles={}
    adapter.runtime=SimpleNamespace(robots=[SimpleNamespace(robot_id=0,position=np.array([.5,.5]),heading=0.)])
    adapter.scheduler=SimpleNamespace(assignments={0:('relay',4)},relay_assignments={4:(0,np.array([2.5,.5]))})
    adapter.agents=[SimpleNamespace(node_coords=np.array([[.5,.5],[.5,.75]]),neighbor_indices=np.array([0,1]))]
    result=adapter._certify_extended_actions([(np.array([2.5,.5]),0.)])
    assert observed.evidence.segment_safe([.5,.5],result[0][0])
    assert not np.array_equal(result[0][0],[2.5,.5])

def test_ordinary_gate_keeps_safety_and_search_authoritative():
    frame,observed=make()
    adapter=MARVELPolicyAdapter.__new__(MARVELPolicyAdapter)
    adapter._observed_belief=observed;adapter._motion_roles={}
    adapter.runtime=SimpleNamespace(robots=[SimpleNamespace(robot_id=0,position=np.array([.5,.5]),heading=0.)])
    for role in ('safety','target_search'):
        adapter.scheduler=SimpleNamespace(assignments={0:(role,4)})
        actions=[(np.array([2.5,.5]),20.)]
        assert adapter._certify_extended_actions(actions)[0] is actions[0]

def test_changed_ordinary_prefix_cannot_bypass_dynamic_warning_buffer():
    frame,observed=make()
    observed.observe(observation(frame,[[0,0,0],[1,0,1]]))
    adapter=MARVELPolicyAdapter.__new__(MARVELPolicyAdapter)
    adapter._observed_belief=observed;adapter._motion_roles={}
    adapter.runtime=SimpleNamespace(current_step=0,robots=[SimpleNamespace(robot_id=0,position=np.array([.5,.5]),heading=0.)])
    adapter.scheduler=SimpleNamespace(assignments={},safety=SimpleNamespace(hazards=SimpleNamespace(
        segment_intersects_warning_buffer=lambda start,end,step:np.linalg.norm(end-start)>.1)))
    adapter.agents=[]
    result=adapter._certify_extended_actions([(np.array([2.5,.5]),0.)])
    np.testing.assert_array_equal(result[0][0],[.5,.5])

def test_arbitrarily_small_positive_overlap_never_certifies_unseen_cell():
    frame=GeometryFrame(0.,0.,1.,3,3,mode='extended')
    belief=ObservedBeliefMap(frame,(0.,0.),.5000000000001,(2,2))
    belief.observe(observation(frame,[[0,0,0]]))
    assert belief.rasterize()[0,1]==UNKNOWN
