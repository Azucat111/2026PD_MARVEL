"""Experiment-only common candidate budget; not a physical feasibility mask."""
import numpy as np

def limit_travel_candidates(graph,budget):
    if not isinstance(budget,int) or isinstance(budget,bool) or budget<1:
        raise ValueError('positive integer candidate budget required')
    result=graph.action_mask.copy()
    for ti in range(graph.num_subtasks):
        legal=np.flatnonzero(~result[ti])
        ranked=sorted(legal,key=lambda ui:(float(graph.edge_features[ti,ui,2]),graph.uav_states[ui].uav_id))
        result[ti,ranked[budget:]]=True
    return result
