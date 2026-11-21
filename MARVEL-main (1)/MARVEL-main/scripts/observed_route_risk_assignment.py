"""Observed residual-cost probe. No extra planning, truth map or hidden targets."""
import math
import numpy as np
from integrations.gppo.task_graph import TaskType
from integrations.gppo.adapter import GPPOAssignment

def route_risk_assignment(graph,estimates,weight=1.):
    if not math.isfinite(weight) or weight<0:raise ValueError('Nonnegative finite uncertainty weight required')
    rows=[]
    for t,task in enumerate(graph.subtasks):
        for u,uav in enumerate(graph.uav_states):
            if graph.action_mask[t,u]:continue
            cost=float(graph.edge_features[t,u,2]);penalty=0.
            detail=estimates.get((int(task.subtask_id),int(uav.uav_id)))
            if task.task_type==TaskType.TARGET_SEARCH and detail is not None:
                residual=float(detail['unknown_residual_lower_bound_m'])
                if not math.isfinite(residual) or residual<0:raise ValueError('Invalid observed route residual')
                penalty=weight*residual/max(float(uav.velocity),1e-6)
            rows.append((-float(task.priority),cost+penalty,int(task.subtask_id),int(uav.uav_id),t,u))
    if not rows:return None
    *_,t,u=min(rows)
    return GPPOAssignment(t*graph.num_uavs+u,t,u,int(graph.subtasks[t].subtask_id),int(graph.uav_states[u].uav_id),0.)
