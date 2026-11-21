from types import SimpleNamespace
import numpy as np
from scripts.completed_task_stationkeeping import stationkeep_completed_tasks

def context(native=False,complete=True):
    robots=[SimpleNamespace(robot_id=i,position=np.array([i+1.,2.]),heading=17.+i) for i in range(4)]
    reset=[]
    scheduler=SimpleNamespace(_search_recovery=None if native else SimpleNamespace(reset_uid=reset.append),runtime=SimpleNamespace(robots=robots,tasks=SimpleNamespace(all_complete=lambda:complete)),search_assignments={10:1},relay_assignments={8:(2,np.array([9.,9.]))},safety=SimpleNamespace(active_uav_ids={3}))
    actions=[(np.array([10.,10.]),90.) for _ in robots]
    return scheduler,actions,reset

def test_completed_required_tasks_only_idle_uav_holds():
    s,a,reset=context();result=stationkeep_completed_tasks(s,a)
    np.testing.assert_equal(result[0][0],s.runtime.robots[0].position)
    assert result[0][1]==17. and reset==[0]
    for i in (1,2,3):assert result[i] is a[i]

def test_pending_required_task_preserves_exploration_fallback():
    s,a,reset=context(complete=False)
    assert stationkeep_completed_tasks(s,a) is a and reset==[]

def test_native_does_not_read_or_change_task_completion():
    s,a,reset=context(native=True)
    s.runtime.tasks.all_complete=lambda:(_ for _ in ()).throw(AssertionError('native completion touched'))
    assert stationkeep_completed_tasks(s,a) is a

def test_stationkeeping_must_run_after_ordinary_prefix_certification():
 from utils.policy_adapter import MARVELPolicyAdapter
 from utils.geometry import GeometryFrame
 from integrations.gppo.search_recovery import ObservedSearchRecovery
 s,a,_=context()
 rec=ObservedSearchRecovery(GeometryFrame(0,0,1,12,12))
 rec.cells={(x,y):0 for x in range(12) for y in range(12)}
 rec.prefixes[0]=np.array([3.,2.]);rec.anchors[0]=np.array([6.,2.])
 s.assignments={1:('target_search',10),2:('relay',8),3:('safety',None)}
 adapter=MARVELPolicyAdapter.__new__(MARVELPolicyAdapter)
 adapter.runtime=s.runtime;adapter.scheduler=s;adapter.agents=[]
 adapter._observed_belief=SimpleNamespace(evidence=rec)
 adapter._motion_roles={0:('exploration',None),**s.assignments}
 early=stationkeep_completed_tasks(s,a)
 certified=adapter._certify_extended_actions(early)
 assert not np.array_equal(certified[0][0],s.runtime.robots[0].position)
 final=stationkeep_completed_tasks(s,certified,ordinary_controller=rec)
 np.testing.assert_equal(final[0][0],s.runtime.robots[0].position)
 assert 0 not in rec.prefixes
