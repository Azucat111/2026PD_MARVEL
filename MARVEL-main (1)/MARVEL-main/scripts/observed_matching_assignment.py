"""Diagnostic matching of current public Search rows, not a production actor."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from integrations.gppo.task_graph import TaskType
from integrations.gppo.adapter import GPPOAssignment
from integrations.rule_scheduling import choose_rule_assignment

def matching_assignment(graph):
    legal=~np.asarray(graph.action_mask,dtype=bool)
    available=[t for t in range(graph.num_subtasks) if legal[t].any()]
    if not available:return None
    priority=max(float(graph.subtasks[t].priority) for t in available)
    rows=[t for t in available if float(graph.subtasks[t].priority)==priority]
    if len(rows)<2 or any(graph.subtasks[t].task_type!=TaskType.TARGET_SEARCH for t in rows):
        return choose_rule_assignment(graph,True)
    costs=np.asarray(graph.edge_features[:,:,2],dtype=float)
    if not np.isfinite(costs).all() or (costs<0).any():raise ValueError('Finite nonnegative observed costs required')
    # Dummy UAVs prioritize maximum feasible cardinality, then minimum cost.
    penalty=(float(costs.max())+1.)*(len(rows)+1)
    matrix=np.full((len(rows),graph.num_uavs+len(rows)),penalty)
    matrix[:,:graph.num_uavs]=np.where(legal[rows],costs[rows],penalty*4)
    rr,uu=linear_sum_assignment(matrix)
    pairs=[(float(costs[rows[r],u]),int(graph.subtasks[rows[r]].subtask_id),int(graph.uav_states[u].uav_id),rows[r],int(u)) for r,u in zip(rr,uu) if u<graph.num_uavs and legal[rows[r],u]]
    if not pairs:return None
    *_,t,u=min(pairs)
    return GPPOAssignment(t*graph.num_uavs+u,t,u,int(graph.subtasks[t].subtask_id),int(graph.uav_states[u].uav_id),0.)
