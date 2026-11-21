"""Observed Search support precedence; uncertainty is never called infeasibility.

A certified route endpoint inside potential sensor/public-heat support earns
admission precedence over unsupported candidates. It does NOT prove visibility,
target presence or future traffic clearance. No supported candidate => no filter.
"""
import math
from .task_graph import TaskType
from .search_deadline import certified_constant_search_capability

def apply_observed_search_support(runtime,graph,route_estimates):
    if getattr(runtime,'geometry_mode',None)!='extended':return []
    profile=getattr(runtime,'config',{}).get('_gppo_profile',{})
    if not all(k in profile for k in ('search_guaranteed_hidden_radius','search_cell_size')):return []
    reasons=[]
    for ti,task in enumerate(graph.subtasks):
        if task.task_type!=TaskType.TARGET_SEARCH:continue
        legal=[ui for ui in range(graph.num_uavs) if not graph.action_mask[ti,ui]]
        supported=[]
        for ui in legal:
            uid=int(graph.uav_states[ui].uav_id)
            capability=certified_constant_search_capability(runtime,uid)
            route=route_estimates.get((int(task.subtask_id),uid),{})
            residual=route.get('unknown_residual_lower_bound_m',math.inf)
            if (capability is None or route.get('status')!='known_route_plus_unknown_residual'
                    or route.get('real_pose_clearance_certified') is not True
                    or not math.isfinite(float(residual)) or residual<0):continue
            allowance=capability[1]+float(profile['search_guaranteed_hidden_radius'])+math.sqrt(2)*float(profile['search_cell_size'])
            if residual<=allowance:supported.append(ui)
        if not supported:continue
        ids=[int(graph.uav_states[ui].uav_id) for ui in supported]
        for ui in legal:
            if ui in supported:continue
            graph.action_mask[ti,ui]=True
            reasons.append({'task_id':int(task.subtask_id),'deferred_uav_id':int(graph.uav_states[ui].uav_id),
                'supported_uav_ids':ids,'reason':'observed_detection_support_precedence',
                'not_a_physical_infeasibility_claim':True})
    return reasons


def observed_support_yields(runtime,graph,route_estimates,active_uids_by_task):
    from copy import copy
    probe=copy(graph);probe.action_mask=graph.action_mask.copy()
    for task_id,uid in active_uids_by_task.items():
        ti=graph.subtask_id_to_index.get(int(task_id));ui=graph.uav_id_to_index.get(int(uid))
        if ti is not None and ui is not None and graph.capability_matrix[ti,ui] and graph.uav_states[ui].alive:
            probe.action_mask[ti,ui]=False # Inspect the current assignment, not a new allocation.
    deferrals=apply_observed_search_support(runtime,probe,route_estimates)
    result=[];reserved=set()
    for reason in deferrals:
        if active_uids_by_task.get(reason['task_id'])!=reason['deferred_uav_id']:continue
        candidates=[uid for uid in reason['supported_uav_ids'] if uid not in reserved]
        if not candidates:continue
        # Limit releases to distinct available alternatives; actual normal GPPO
        # inference still owns the subsequent assignment.
        reserved.add(min(candidates))
        result.append(reason)
    return result
