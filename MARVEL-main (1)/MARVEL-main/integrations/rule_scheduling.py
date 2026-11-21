"""Conventional priority/nearest-observed-time baseline.

Consumes the same public TaskGraph and hard masks as GPPO. This module neither
loads maps/targets nor changes task lifecycle, motion or capacity.
"""
from __future__ import annotations
import numpy as np
from integrations.gppo.adapter import GPPOAssignment

def choose_rule_assignment(graph, deterministic=True):
    """Priority first; shortest raw total time; stable public-ID tie breaking."""
    if not graph.num_uavs or not graph.num_subtasks:
        return None
    candidates=[]
    for ti,ui in np.argwhere(~np.asarray(graph.action_mask,dtype=bool)):
        ti,ui=int(ti),int(ui)
        task,uav=graph.subtasks[ti],graph.uav_states[ui]
        total=float(graph.edge_features[ti,ui,2])
        if not np.isfinite(total):
            continue
        candidates.append((-float(task.priority),total,int(task.subtask_id),
            int(uav.uav_id),ti,ui))
    if not candidates:
        return None
    _,_,task_id,uav_id,ti,ui=min(candidates)
    return GPPOAssignment(flat_action=int(graph.flatten_action(ti,ui)),
        task_index=ti,uav_index=ui,subtask_id=task_id,uav_id=uav_id,value=0.)
