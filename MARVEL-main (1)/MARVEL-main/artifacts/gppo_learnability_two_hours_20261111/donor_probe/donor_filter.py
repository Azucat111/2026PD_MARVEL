"""One development probe: preserve a Search donor's observed component."""
import copy
import numpy as np
from integrations.gppo.mesh_relay import mesh_components
from integrations.gppo.task_graph import TaskType

def filter_search_donors(graph,comm_range=20.):
 if float(comm_range)!=20.:raise ValueError('Frozen observed and physical radio range must be20m')
 original=np.asarray(graph.action_mask,dtype=bool).copy();filtered=original.copy();positions={};missing=[]
 for u in graph.uav_states:
  p=None if u.position is None else np.asarray(u.position,dtype=float)
  if p is None or p.shape!=(2,) or not np.isfinite(p).all():missing.append(int(u.uav_id))
  else:positions[int(u.uav_id)]=p
 row=dict(time=float(graph.current_time),uav_ids=[int(u.uav_id) for u in graph.uav_states],public_positions={uid:p.tolist() for uid,p in positions.items()},original_mask=original.tolist(),filtered_mask=original.tolist(),blocked_ids=[],missing_public_position_ids=missing,fallback=False,applied=False)
 if not graph.subtasks or not all(int(t.task_type)==int(TaskType.TARGET_SEARCH) for t in graph.subtasks):row['reason']='not_pure_search';return graph,row
 if missing:row['reason']='missing_public_positions_no_guess';return graph,row
 components=mesh_components(positions,comm_range);blocked=[]
 for ui,u in enumerate(graph.uav_states):
  uid=int(u.uav_id)
  if original[:,ui].all():continue
  donor=next(c for c in components if uid in c)
  remaining={v:positions[v] for v in donor if v!=uid}
  if len(mesh_components(remaining,comm_range))>1:filtered[:,ui]=True;blocked.append(uid)
 row['blocked_ids']=blocked;row['applied']=True;row['reason']='donor_component_removal_only'
 if filtered.all():filtered=original.copy();row['fallback']=True;row['reason']='no_legal_action_fallback_original'
 row['filtered_mask']=filtered.tolist();result=copy.copy(graph);result.action_mask=filtered.copy();return result,row
