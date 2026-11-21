from types import SimpleNamespace
import numpy as np
from utils.geometry import GeometryFrame
from utils.policy_adapter import MARVELPolicyAdapter
from integrations.gppo.search_recovery import ObservedSearchRecovery

def make_adapter(token='relay',authoritative=True):
    robot=SimpleNamespace(robot_id=0,position=np.array([2.,2.]),heading=17.)
    ordinary=ObservedSearchRecovery(GeometryFrame(0,0,1,12,12));ordinary.cells={(x,y):0 for x in range(12) for y in range(12)}
    ordinary.prefixes[0]=np.array([4.,2.]);ordinary.anchors[0]=np.array([6.,2.])
    scheduler=SimpleNamespace(assignments={0:(token,7)},relay_assignments={7:(0,np.array([6.,2.]))},search_assignments={},safety=SimpleNamespace(active_uav_ids=set()),_search_recovery=SimpleNamespace(reset_uid=lambda uid:None) if authoritative else None)
    adapter=MARVELPolicyAdapter.__new__(MARVELPolicyAdapter);adapter.runtime=SimpleNamespace(robots=[robot],tasks=SimpleNamespace(all_complete=lambda:True));adapter.scheduler=scheduler
    adapter._observed_belief=SimpleNamespace(evidence=ordinary);adapter.agents=[];adapter._motion_roles={0:(token,7)}
    return adapter,ordinary

def test_relay_own_recovery_hold_is_not_replaced_by_ordinary_cached_prefix():
    adapter,_=make_adapter();action=(np.array([2.,2.]),17.)
    result=adapter._certify_extended_actions([action])
    np.testing.assert_equal(result[0][0],action[0]);assert result[0][1]==17.

def test_legacy_relay_without_authoritative_prefix_still_gets_certified():
    adapter,_=make_adapter(authoritative=False);action=(np.array([2.,2.]),17.)
    result=adapter._certify_extended_actions([action])
    assert not np.array_equal(result[0][0],action[0])

def test_final_stationkeeping_clears_ordinary_prefix_only_after_required_tasks():
    adapter,ordinary=make_adapter(token='exploration')
    adapter.scheduler.relay_assignments={}
    a=[(np.array([4.,2.]),90.)]
    result=adapter._stationkeep_completed_tasks(a)
    np.testing.assert_equal(result[0][0],adapter.runtime.robots[0].position)
    assert result[0][1]==17. and 0 not in ordinary.prefixes
    adapter.runtime.tasks.all_complete=lambda:False
    assert adapter._stationkeep_completed_tasks(a) is a

def test_native_stationkeeping_never_reads_task_status():
    adapter,_=make_adapter();adapter._observed_belief=None
    adapter.runtime.tasks.all_complete=lambda:(_ for _ in ()).throw(AssertionError('native status read'))
    a=[(np.array([4.,2.]),90.)]
    assert adapter._stationkeep_completed_tasks(a) is a


def test_stationkeeping_preserves_search_relay_and_safety_commands():
    for role in ('search','relay','safety'):
        adapter,_=make_adapter(token='exploration')
        adapter.scheduler.relay_assignments={}
        if role=='search':adapter.scheduler.search_assignments={7:0}
        elif role=='relay':adapter.scheduler.relay_assignments={7:(0,np.array([6.,2.]))}
        else:adapter.scheduler.safety.active_uav_ids={0}
        a=[(np.array([4.,2.]),90.)]
        result=adapter._stationkeep_completed_tasks(a)
        np.testing.assert_equal(result[0][0],a[0][0]);assert result[0][1]==90.
