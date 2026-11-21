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
class LinearFindAssignment:
 def __init__(self,runtime,mode,minimum_gain=.5):
  self.runtime=runtime;self.minimum_gain=minimum_gain;self.rows=[];self.requires_graph_embedding=False
  if mode=='nominal':
   self.regressor=None;self.artifact={'feature_count':184}
  else:
   assert mode=='linear'
   p=Path(__file__).resolve().parents[1]/'artifacts/extended_linear_find_cost_preflight_20261015_attempt29/linear_public8_alpha10.joblib'
   assert hashlib.sha256(p.read_bytes()).hexdigest()=='78dc63ecdd4b4d8ca9c4fc223a4747a0220b430f9da490c77a655d662e2ed8e0'
   self.artifact=joblib.load(p);self.regressor=self.artifact['estimator']
   assert not self.artifact['requires_graph_embedding']
 def choose(self,graph,deterministic=True):
  baseline=choose_rule_assignment(graph,deterministic)
  if baseline is None or graph.subtasks[baseline.task_index].task_type!=TaskType.TARGET_SEARCH:return baseline
  scheduler=self.runtime.high_level_scheduler;observer=scheduler.allocator.adapter.heading_observations
  observer.public_search_context=public_search_context(scheduler);batch=opportunity_batch(graph,observer,'cpu')
  priority=float(graph.subtasks[baseline.task_index].priority)
  pairs=[(ti,ui) for ti,ui in np.argwhere(~graph.action_mask) if graph.subtasks[int(ti)].task_type==TaskType.TARGET_SEARCH and float(graph.subtasks[int(ti)].priority)==priority]
  actions=[int(ti)*graph.num_uavs+int(ui) for ti,ui in pairs]
  representations=np.zeros((len(actions),128),dtype=float) # Selected public8 head does not consume graph-embedding columns.
  raw=dict(raw_uav_features=graph.uav_features,raw_task_features=graph.task_features,raw_edge_features=graph.edge_features,mask=graph.action_mask)
  edges=batch['edge_features'][0].numpy();dt=self.runtime.dt;hazards=self.runtime.config.get('dynamic_obstacles',[])
  features=np.asarray([find_cost_features(raw,edges,int(ti),int(ui),graph.current_time,hazards,dt,z) for (ti,ui),z in zip(pairs,representations)])
  assert features.shape[1]==self.artifact['feature_count']
  prediction=np.clip(features[:,34]*80. if self.regressor is None else self.regressor.predict(features),dt,150.)
  cost=np.asarray(graph.edge_features[:,:,2],dtype=float).copy()
  for (ti,ui),value in zip(pairs,prediction):cost[ti,ui]=value
  shadow=copy(graph);shadow.edge_features=graph.edge_features.copy();shadow.edge_features[:,:,2]=cost+1e-4*np.asarray(graph.edge_features[:,:,2],dtype=float)
  selected=matching_assignment(shadow)
  if selected is None or graph.subtasks[selected.task_index].task_type!=TaskType.TARGET_SEARCH:selected=baseline
  if cost[baseline.task_index,baseline.uav_index]-cost[selected.task_index,selected.uav_index]<self.minimum_gain:selected=baseline
  assert not graph.action_mask[selected.task_index,selected.uav_index]
  self.rows.append(dict(step=int(self.runtime.current_step),baseline=dict(task=baseline.subtask_id,uid=baseline.uav_id),selected=dict(task=selected.subtask_id,uid=selected.uav_id),predicted_baseline_seconds=float(cost[baseline.task_index,baseline.uav_index]),predicted_selected_seconds=float(cost[selected.task_index,selected.uav_index]),actor_batch={k:v.cpu().tolist() for k,v in batch.items()},chosen_action=int(selected.flat_action),uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks]))
  return selected
