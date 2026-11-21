import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
from filter_scheduler_candidates import limit_travel_candidates

def test_limits_only_currently_legal_pairs_and_keeps_cheapest():
    state=TaskGraph([UAVState(i) for i in range(6)],[Subtask(7,TaskType.TARGET_SEARCH,5)],travel_time_matrix=np.array([[6.,1.,2.,3.,4.,5.]]))
    state.action_mask[0,1]=True
    result=limit_travel_candidates(state,3)
    assert np.flatnonzero(~result[0]).tolist()==[2,3,4]
    assert state.action_mask[0,0]==False # Original mask is not silently overwritten.

def test_masked_context_and_id_ties_are_stable():
    state=TaskGraph([UAVState(9),UAVState(2),UAVState(5)],
        [Subtask(7,TaskType.TARGET_SEARCH,5),Subtask(8,TaskType.RELAY,2)],travel_time_matrix=np.ones((2,3)))
    state.action_mask[1,:]=True
    result=limit_travel_candidates(state,2)
    assert np.flatnonzero(~result[0]).tolist()==[1,2]
    assert result[1].all()
