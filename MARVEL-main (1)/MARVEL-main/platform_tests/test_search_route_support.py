from types import SimpleNamespace
import numpy as np
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
from integrations.gppo.route_support import apply_observed_search_support
from utils.dynamics_models import KinematicSimple
from utils.sensor_models import IdealSensor

def runtime(mode='extended'):
    return SimpleNamespace(geometry_mode=mode,
        robots=[SimpleNamespace(robot_id=i,robot_type='explorer') for i in (0,1)],
        dynamics=KinematicSimple({'velocity':1}),dynamics_by_type={},sensor_by_type={'explorer':IdealSensor({'range':10})},
        config={'_gppo_profile':{'search_guaranteed_hidden_radius':5.5,'search_cell_size':1}})

def graph():
    return TaskGraph([UAVState(0),UAVState(1)],
        [Subtask(7,TaskType.TARGET_SEARCH,5),Subtask(8,TaskType.RELAY,2)])

def estimates(a,b):
    return {(7,0):{'status':'known_route_plus_unknown_residual','real_pose_clearance_certified':True,'unknown_residual_lower_bound_m':a},
        (7,1):{'status':'known_route_plus_unknown_residual','real_pose_clearance_certified':True,'unknown_residual_lower_bound_m':b}}

def test_prefer_observed_detection_support_without_discarding_encoder_context():
    g=graph();before=g.model_uav_features.copy()
    reasons=apply_observed_search_support(runtime(),g,estimates(50,12))
    assert g.action_mask.tolist()==[[True,False],[False,False]]
    assert reasons[0]['deferred_uav_id']==0 and reasons[0]['supported_uav_ids']==[1]
    assert np.array_equal(g.model_uav_features,before)

def test_no_supported_candidate_keeps_uncertain_candidates_available():
    g=graph();assert apply_observed_search_support(runtime(),g,estimates(50,40))==[]
    assert not g.action_mask.any()

def test_unavailable_supported_peer_is_not_an_alternative():
    g=graph();g.action_mask[0,1]=True
    assert apply_observed_search_support(runtime(),g,estimates(50,12))==[]
    assert not g.action_mask[0,0]

def test_native_and_missing_route_evidence_preserve_original_masks():
    g=graph();assert apply_observed_search_support(runtime('marvel_native'),g,estimates(50,12))==[]
    assert not g.action_mask.any()
    assert apply_observed_search_support(runtime(),g,{})==[]


def test_uncertified_graph_only_route_cannot_earn_precedence():
    g=graph();routes=estimates(50,12);routes[(7,1)]['real_pose_clearance_certified']=False
    assert apply_observed_search_support(runtime(),g,routes)==[]
    assert not g.action_mask.any()

from integrations.gppo.route_support import observed_support_yields
from integrations.gppo.event_allocator import GPPOEventAllocator

def test_existing_unsupported_search_can_yield_only_to_available_supported_peer():
    g=graph();g.action_mask[0,0]=True
    result=observed_support_yields(runtime(),g,estimates(50,12),{7:0})
    assert len(result)==1 and result[0]['deferred_uav_id']==0
    assert g.action_mask[0,0] # Original unavailable mask remains unchanged.
    g.action_mask[0,1]=True
    assert observed_support_yields(runtime(),g,estimates(50,12),{7:0})==[]

def test_supported_active_search_is_not_preempted():
    g=graph();g.action_mask[0,0]=True
    assert observed_support_yields(runtime(),g,estimates(12,13),{7:0})==[]

def test_allocator_route_support_binding_keeps_native_unchanged():
    g=graph();a=GPPOEventAllocator.__new__(GPPOEventAllocator);a.runtime=runtime('marvel_native')
    a.last_travel_estimates=estimates(50,12);a._apply_route_support_mask(g)
    assert not g.action_mask.any()
    a.runtime=runtime();a._apply_route_support_mask(g)
    assert g.action_mask[0].tolist()==[True,False]
