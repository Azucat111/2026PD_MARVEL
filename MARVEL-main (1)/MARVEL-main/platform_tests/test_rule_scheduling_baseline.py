"""A fair conventional scheduling decision on the same public TaskGraph."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.gppo.task_graph import TaskGraph,Subtask,TaskType,UAVState
from integrations.rule_scheduling import choose_rule_assignment

def graph(uavs=None,tasks=None,times=None):
    uavs=uavs or [UAVState(14),UAVState(22)]
    tasks=tasks or [Subtask(3,TaskType.TARGET_SEARCH,2),Subtask(11,TaskType.TARGET_SEARCH,5)]
    if times is None:times=np.array([[1.,2.],[4.,3.]])
    return TaskGraph(uavs,tasks,current_time=0,travel_time_matrix=times)

def test_rule_uses_priority_then_observed_raw_total_time():
    result=choose_rule_assignment(graph())
    assert (result.subtask_id,result.uav_id)==(11,22)
    assert result.flat_action==3

def test_rule_respects_event_masks_even_for_best_priority():
    state=graph()
    state.action_mask[1,:]=True
    result=choose_rule_assignment(state)
    assert (result.subtask_id,result.uav_id)==(3,14)

def test_rule_uses_real_capability_busy_and_alive_masks():
    uavs=[UAVState(0,can_search=False),UAVState(1,available=False),
        UAVState(2,busy_until=5),UAVState(3,alive=False),UAVState(9)]
    state=graph(uavs=uavs,tasks=[Subtask(7,TaskType.TARGET_SEARCH,5)],
        times=np.array([[0.,0.,0.,0.,20.]]))
    assert choose_rule_assignment(state).uav_id==9

def test_rule_ties_follow_ids_not_input_row_order():
    tasks=[Subtask(8,TaskType.TARGET_SEARCH,5),Subtask(3,TaskType.TARGET_SEARCH,5)]
    state=graph(uavs=[UAVState(22),UAVState(14)],tasks=tasks,times=np.ones((2,2)))
    decision=choose_rule_assignment(state)
    assert (decision.subtask_id,decision.uav_id)==(3,14)

def test_rule_returns_none_when_every_action_masked():
    state=graph()
    state.action_mask[:]=True
    assert choose_rule_assignment(state) is None

def test_real_allocator_reserves_resources_between_rule_decisions(monkeypatch):
    from types import SimpleNamespace
    import integrations.gppo.event_allocator as module
    from integrations.gppo.event_allocator import GPPOEventAllocator,GPPOEventSlot
    base=graph(uavs=[UAVState(14,position=(0.,0.)),UAVState(22,position=(10.,0.))],
        tasks=[Subtask(99,TaskType.EXPLORATION,3)],times=np.ones((1,2)))
    monkeypatch.setattr(module,'RuntimeGraphBuilder',lambda *args,**kwargs:SimpleNamespace(build=lambda:SimpleNamespace(graph=base)))
    allocator=GPPOEventAllocator.__new__(GPPOEventAllocator)
    allocator.runtime=SimpleNamespace(current_step=0,dt=.1)
    seen=[]
    def assign(state,deterministic=True):
        seen.append(state.action_mask.copy())
        return choose_rule_assignment(state,deterministic)
    allocator.adapter=SimpleNamespace(assign=assign)
    allocator._marvel_agents={}
    slots=[GPPOEventSlot(3,TaskType.TARGET_SEARCH,5,position=(0.,0.),processing_time=6),
           GPPOEventSlot(11,TaskType.TARGET_SEARCH,5,position=(1.,0.),processing_time=6)]
    assigned=allocator.allocate(slots)
    assert [(x.slot_id,x.uav_id) for x in assigned]==[(3,14),(11,22)]
    assert seen[1][:,0].all()

def test_real_allocator_context_and_self_relay_masks_reach_rule(monkeypatch):
    from types import SimpleNamespace
    import integrations.gppo.event_allocator as module
    from integrations.gppo.event_allocator import GPPOEventAllocator,GPPOEventSlot
    base=graph(uavs=[UAVState(14,position=(0.,0.)),UAVState(22,position=(10.,0.))],
        tasks=[Subtask(99,TaskType.EXPLORATION,3)],times=np.ones((1,2)))
    monkeypatch.setattr(module,'RuntimeGraphBuilder',lambda *args,**kwargs:SimpleNamespace(build=lambda:SimpleNamespace(graph=base)))
    allocator=GPPOEventAllocator.__new__(GPPOEventAllocator)
    allocator.runtime=SimpleNamespace(current_step=0,dt=.1)
    allocator.adapter=SimpleNamespace(assign=choose_rule_assignment);allocator._marvel_agents={}
    slots=[GPPOEventSlot(3,TaskType.TARGET_SEARCH,5,position=(0.,0.),forbidden_uav_ids=(14,))]
    rows=[Subtask(99,TaskType.RELAY,100,position=(0.,0.)),Subtask(3,TaskType.TARGET_SEARCH,5,position=(0.,0.))]
    assigned=allocator.allocate(slots,context_subtasks=rows)
    assert [(x.slot_id,x.uav_id) for x in assigned]==[(3,22)]
