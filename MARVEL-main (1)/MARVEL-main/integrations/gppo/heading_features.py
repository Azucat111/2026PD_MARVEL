"""Experimental heading features derived exclusively from public sensor snapshots.
This schema is separate from the frozen/native graph protocol.
"""
import math
import numpy as np
import torch
from integrations.gppo.gppo_model import GPPOActorCritic,batch_single_task_graph

HEADING_PROTOCOL='extended-gppo-observed-heading-v2'

class HeadingObservations:
    def __init__(self,age_fraction=.05):
        self.age_fraction=float(age_fraction)
        if not 0<=self.age_fraction<=1:raise ValueError('Invalid public-state age fraction')
        self.previous={};self.current={}

    def observe(self,observations):
        self.previous=self.current.copy();self.current={}
        for uid,observation in observations.items():
            heading=observation.get('heading')
            if heading is not None and math.isfinite(float(heading)):
                self.current[int(uid)]=float(heading)%360.

    def heading(self,uid):
        if uid not in self.current:return None
        current=self.current[uid];previous=self.previous.get(uid,current)
        delta=(previous-current+180.)%360.-180.
        return (current+self.age_fraction*delta)%360.

def heading_batch(graph,observations,device):
    batch=batch_single_task_graph(graph,device=device)
    extra=np.zeros((graph.num_subtasks,graph.num_uavs,3),dtype=np.float32)
    for t,task in enumerate(graph.subtasks):
        if task.position is None:continue
        for u,uav in enumerate(graph.uav_states):
            angle=observations.heading(int(uav.uav_id))
            if angle is None:continue
            delta=np.asarray(task.position)-np.asarray(uav.position)
            if not np.isfinite(delta).all():raise ValueError('Nonfinite public task pose')
            bearing=math.atan2(float(delta[1]),float(delta[0]))
            relative=bearing-math.radians(angle)
            extra[t,u]=[math.cos(relative),math.sin(relative),1.]
    batch['edge_features']=torch.cat([batch['edge_features'],
        torch.as_tensor(extra,device=device).unsqueeze(0)],dim=-1)
    return batch

def initialize_heading_model(state,device):
    model=GPPOActorCritic(edge_feature_dim=6).to(device)
    target=model.state_dict()
    if set(state)!=set(target):raise ValueError('Incomplete or extra warm-start parameters')
    for key,value in state.items():
        if key in ('ahgnn.edge_input.weight','actor.edge_proj.weight'):
            if value.shape[1]!=3:raise ValueError('Warm start must have original three edge features')
            target[key].zero_();target[key][:,:3].copy_(value)
        else:target[key].copy_(value)
    model.load_state_dict(target,strict=True);model.eval()
    return model
