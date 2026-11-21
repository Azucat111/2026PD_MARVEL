"""Explicit extended public opportunity inputs; no environmental truth access."""
import numpy as np
import torch
from .heading_features import heading_batch
from .gppo_model import GPPOActorCritic
from .task_graph import TaskType
OPPORTUNITY_PROTOCOL='extended-gppo-public-opportunity-v1'


def nominal_find_eta(position,prior,speed,sensor_range):
 position=np.asarray(position,dtype=float);prior=np.asarray(prior,dtype=float)
 if position.shape!=(2,) or prior.shape!=(2,) or not np.isfinite(position).all() or not np.isfinite(prior).all():raise ValueError('Finite public positions required')
 if not np.isfinite(speed) or not np.isfinite(sensor_range) or sensor_range<0:raise ValueError('Finite public speed/range required')
 remaining=max(0.,float(np.linalg.norm(prior-position))-sensor_range)
 return min(80.,remaining/max(float(speed),1e-6))


def opportunity_extra(graph,context):
 extra=np.zeros((graph.num_subtasks,graph.num_uavs,3),dtype=np.float32)
 sensor_range=10. if context is None else float(context['sensor_range'])
 priors=[] if context is None else context['pending_priors']
 reserved=set() if context is None else set(map(int,context['reserved_uids']))
 # Validate even if no UAV is eligible; never silently accept hidden bad data.
 for _hid,prior in priors:nominal_find_eta((0.,0.),prior,1.,sensor_range)
 for ti,task in enumerate(graph.subtasks):
  relevant=[(hid,p) for hid,p in priors if not (task.task_type==TaskType.TARGET_SEARCH and int(task.subtask_id)==100000+int(hid))]
  eligible=[u for u in graph.uav_states if u.alive and u.available and u.can_search and int(u.uav_id) not in reserved and u.assigned_task_num==0]
  costs=[[nominal_find_eta(u.position,p,u.velocity,sensor_range) for u in eligible] for _,p in relevant]
  for ui,uav in enumerate(graph.uav_states):
   if task.task_type==TaskType.TARGET_SEARCH and task.position is not None:
    extra[ti,ui,0]=nominal_find_eta(uav.position,task.position,uav.velocity,sensor_range)/80.
   if context is None:continue
   extra[ti,ui,2]=1.
   regrets=[]
   for values in costs:
    before=min(values,default=80.)
    after=min((v for u,v in zip(eligible,values) if int(u.uav_id)!=int(uav.uav_id)),default=80.)
    regrets.append(max(0.,after-before))
   extra[ti,ui,1]=(float(np.mean(regrets)) if regrets else 0.)/80.
 return extra


def opportunity_batch(graph,observations,device):
 batch=heading_batch(graph,observations,device)
 context=getattr(observations,'public_search_context',None)
 extra=opportunity_extra(graph,context)
 batch['edge_features']=torch.cat([batch['edge_features'],torch.as_tensor(extra,device=device).unsqueeze(0)],dim=-1)
 return batch


def public_search_context(scheduler):
 # heat_by_id is explicitly installed from the public scenario projection.
 assigned=getattr(scheduler,'search_assignments',{})
 serviced=set(getattr(scheduler,'serviced_heat_ids',set()))
 reserved=set(map(int,assigned.values()))|{int(value[0]) for value in getattr(scheduler,'relay_assignments',{}).values()}
 reserved|=set(map(int,getattr(getattr(scheduler,'safety',None),'active_uav_ids',set())))
 pending=[[int(h),list(p.position)] for h,p in sorted(getattr(scheduler,'heat_by_id',{}).items()) if int(h) not in serviced and int(h) not in assigned]
 return dict(pending_priors=pending,reserved_uids=sorted(reserved),sensor_range=float(scheduler.runtime.sensor.sensor_range))


def initialize_opportunity_model(state,device):
 model=GPPOActorCritic(edge_feature_dim=9).to(device);target=model.state_dict()
 if set(state)!=set(target):raise ValueError('Complete heading-v2 parameters required')
 with torch.no_grad():
  for key,value in state.items():
   if key in ('ahgnn.edge_input.weight','actor.edge_proj.weight'):
    if value.shape[1]!=6:raise ValueError('Warm start must have six heading edge features')
    target[key].zero_();target[key][:,:6].copy_(value)
   else:target[key].copy_(value)
 model.load_state_dict(target,strict=True);model.eval();return model
