from types import SimpleNamespace as S
import numpy as np
from scripts.observed_matching_assignment import matching_assignment
from integrations.gppo.task_graph import TaskType

def graph(costs,mask=None,priorities=None):
    c=np.asarray(costs,dtype=float);t,u=c.shape
    return S(flatten_action=lambda ti,ui:ti*u+ui,num_subtasks=t,num_uavs=u,action_mask=np.zeros_like(c,dtype=bool) if mask is None else np.asarray(mask,dtype=bool),edge_features=np.stack([c,np.zeros_like(c),c],axis=-1),subtasks=[S(priority=(priorities or [5]*t)[i],task_type=TaskType.TARGET_SEARCH,subtask_id=100+i) for i in range(t)],uav_states=[S(uav_id=i) for i in range(u)])

def test_matching_preserves_uav_needed_by_other_current_search():
    a=matching_assignment(graph([[1,2],[1.1,100]]))
    assert (a.task_index,a.uav_index)==(1,0)

def test_matching_never_selects_masked_pair():
    a=matching_assignment(graph([[1,2],[1.1,100]],[[False,False],[True,False]]))
    assert (a.task_index,a.uav_index)==(0,0)

def test_priority_precedes_joint_cost():
    a=matching_assignment(graph([[100,101],[1,2]],priorities=[6,5]))
    assert (a.task_index,a.uav_index)==(0,0)

def test_more_rows_than_available_uavs_maximizes_feasible_cardinality():
    a=matching_assignment(graph([[3],[1],[2]]))
    assert (a.task_index,a.uav_index)==(1,0)

def test_fully_masked_graph_returns_none():
    assert matching_assignment(graph([[1]],[[True]])) is None
