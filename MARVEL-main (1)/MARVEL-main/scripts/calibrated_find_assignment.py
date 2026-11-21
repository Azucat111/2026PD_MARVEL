"""Learned Find-time head over the existing frozen GPPO graph representation."""
from copy import copy
from pathlib import Path
import hashlib,json
import numpy as np
import torch,joblib
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.opportunity_features import opportunity_batch,public_search_context
from integrations.gppo.task_graph import TaskType
from integrations.rule_scheduling import choose_rule_assignment
from scripts.observed_matching_assignment import matching_assignment
from scripts.paired_gain_calibration import candidate_representations
from scripts.observed_find_cost_features import find_cost_features,FEATURE_PROTOCOL
class CalibratedFindAssignment:
 def __init__(self,runtime,candidate_path,expected_sha256,minimum_gain=.5):
  p=Path(candidate_path);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected_sha256
  artifact=joblib.load(p);assert artifact['feature_protocol']==FEATURE_PROTOCOL
  self.runtime=runtime;self.artifact=artifact;self.regressor=artifact['estimator'];self.minimum_gain=minimum_gain;self.rows=[]
  warm=Path(artifact['frozen_encoder']);assert hashlib.sha256(warm.read_bytes()).hexdigest()==artifact['frozen_encoder_sha256']
  cp=torch.load(warm,map_location='cpu',weights_only=False);self.model=GPPOActorCritic(edge_feature_dim=9);self.model.load_state_dict(cp['model']);self.model.eval()
  for parameter in self.model.parameters():parameter.requires_grad_(False)
 def choose(self,graph,deterministic=True):
  baseline=choose_rule_assignment(graph,deterministic)
  if baseline is None or graph.subtasks[baseline.task_index].task_type!=TaskType.TARGET_SEARCH:return baseline
  scheduler=self.runtime.high_level_scheduler;observer=scheduler.allocator.adapter.heading_observations
  observer.public_search_context=public_search_context(scheduler);batch=opportunity_batch(graph,observer,'cpu')
  priority=float(graph.subtasks[baseline.task_index].priority)
  pairs=[(ti,ui) for ti,ui in np.argwhere(~graph.action_mask) if graph.subtasks[int(ti)].task_type==TaskType.TARGET_SEARCH and float(graph.subtasks[int(ti)].priority)==priority]
  actions=[int(ti)*graph.num_uavs+int(ui) for ti,ui in pairs]
  representations=candidate_representations(self.model,batch,actions)
  raw=dict(raw_uav_features=graph.uav_features,raw_task_features=graph.task_features,raw_edge_features=graph.edge_features,mask=graph.action_mask)
  edges=batch['edge_features'][0].numpy();dt=self.runtime.dt;hazards=self.runtime.config.get('dynamic_obstacles',[])
  features=np.asarray([find_cost_features(raw,edges,int(ti),int(ui),graph.current_time,hazards,dt,z) for (ti,ui),z in zip(pairs,representations)])
  assert features.shape[1]==self.artifact['feature_count']
  prediction=np.clip(self.regressor.predict(features),dt,150.)
  cost=np.asarray(graph.edge_features[:,:,2],dtype=float).copy()
  for (ti,ui),value in zip(pairs,prediction):cost[ti,ui]=value
  shadow=copy(graph);shadow.edge_features=graph.edge_features.copy();shadow.edge_features[:,:,2]=cost
  selected=matching_assignment(shadow)
  if selected is None or graph.subtasks[selected.task_index].task_type!=TaskType.TARGET_SEARCH:selected=baseline
  if cost[baseline.task_index,baseline.uav_index]-cost[selected.task_index,selected.uav_index]<self.minimum_gain:selected=baseline
  assert not graph.action_mask[selected.task_index,selected.uav_index]
  self.rows.append(dict(step=int(self.runtime.current_step),baseline=dict(task=baseline.subtask_id,uid=baseline.uav_id),selected=dict(task=selected.subtask_id,uid=selected.uav_id),predicted_baseline_seconds=float(cost[baseline.task_index,baseline.uav_index]),predicted_selected_seconds=float(cost[selected.task_index,selected.uav_index]),actor_batch={k:v.cpu().tolist() for k,v in batch.items()},chosen_action=int(selected.flat_action),uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks]))
  return selected
