"""Diagnostic teachers reuse exact sensor geometry and existing SciPy matching."""
from copy import copy
import numpy as np
from utils.sensor_models import IdealSensor
from utils.geometry import iter_cells_in_radius
from integrations.gppo.task_graph import TaskType
from integrations.gppo.adapter import GPPOAssignment
from integrations.rule_scheduling import choose_rule_assignment
from scripts.observed_matching_assignment import matching_assignment
class RouteVisibilityTeacher:
 def __init__(self,runtime,mode,minimum_gain=.1):
  assert mode in ['greedy','matching'];self.runtime=runtime;self.mode=mode;self.minimum_gain=minimum_gain;self.rows=[];self.tick=None
 def choose(self,graph,deterministic=True):
  baseline=choose_rule_assignment(graph,deterministic)
  if baseline is None or graph.subtasks[baseline.task_index].task_type!=TaskType.TARGET_SEARCH:return baseline
  scheduler=self.runtime.high_level_scheduler;rec=scheduler._search_recovery;views=scheduler._search_viewpoints
  observer=scheduler.allocator.adapter.heading_observations
  if self.tick!=self.runtime.current_step:
   self.tick=self.runtime.current_step;self.visible={};self.priors={}
   self.known=dict(rec.cells);self.potential=np.zeros(rec.frame.shape,dtype=np.uint8)
   for (c,r),v in self.known.items():
    if v==1:self.potential[r,c]=1
  fractions=np.zeros((graph.num_subtasks,graph.num_uavs),dtype=float);eligible=[]
  robot_types={int(r.robot_id):r.robot_type for r in self.runtime.robots}
  priority=float(graph.subtasks[baseline.task_index].priority)
  for ti,task in enumerate(graph.subtasks):
   if task.task_type!=TaskType.TARGET_SEARCH or float(task.priority)!=priority:continue
   if task.position is None:continue
   heat=int(task.subtask_id)-100000
   if heat not in self.priors:
    checked=views.checked_cells_by_heat.get(heat,set());anchor=np.asarray(task.position)
    self.priors[heat]={tuple(map(int,c)) for c in iter_cells_in_radius(rec.frame,anchor,views.prior_radius)
     if self.known.get(tuple(map(int,c)))!=1 and tuple(map(int,c)) not in checked and np.linalg.norm(rec.frame.cell_to_world(c)-anchor)<=views.prior_radius}
   prior=self.priors[heat]
   if not prior:continue
   for ui,uav in enumerate(graph.uav_states):
    if graph.action_mask[ti,ui]:continue
    angle=observer.heading(int(uav.uav_id))
    if angle is None:continue
    fractions[ti,ui]=self.preview_fraction(graph,ti,ui,prior)
    eligible.append((ti,ui))
  if not eligible:return baseline
  if self.mode=='greedy':
   ti,ui=min(eligible,key=lambda x:(-fractions[x],float(graph.edge_features[x[0],x[1],2]),int(graph.subtasks[x[0]].subtask_id),int(graph.uav_states[x[1]].uav_id)))
   selected=GPPOAssignment(ti*graph.num_uavs+ui,ti,ui,int(graph.subtasks[ti].subtask_id),int(graph.uav_states[ui].uav_id),0.)
  else:
   shadow=copy(graph);shadow.edge_features=graph.edge_features.copy()
   shadow.edge_features[:,:,2]=(1.-fractions)*80.+np.nan_to_num(graph.edge_features[:,:,2],nan=1e6,posinf=1e6)*.01
   selected=matching_assignment(shadow)
   if selected is None or graph.subtasks[selected.task_index].task_type!=TaskType.TARGET_SEARCH:selected=baseline
  if fractions[selected.task_index,selected.uav_index]<fractions[baseline.task_index,baseline.uav_index]+self.minimum_gain:selected=baseline
  assert not graph.action_mask[selected.task_index,selected.uav_index]
  self.rows.append(dict(step=int(self.runtime.current_step),baseline=dict(task=baseline.subtask_id,uid=baseline.uav_id),selected=dict(task=selected.subtask_id,uid=selected.uav_id),baseline_fraction=float(fractions[baseline.task_index,baseline.uav_index]),selected_fraction=float(fractions[selected.task_index,selected.uav_index])))
  return selected

 def preview_fraction(self,graph,ti,ui,prior):
  from copy import copy
  from types import SimpleNamespace
  import pickle,hashlib
  from utils.dynamics_models import KinematicSimple
  from integrations.gppo.observed_graph import observed_next_hop
  s=self.runtime.high_level_scheduler;source=s._search_recovery;task=graph.subtasks[ti];uav=graph.uav_states[ui];uid=int(uav.uav_id)
  angle=s.allocator.adapter.heading_observations.heading(uid)
  if angle is None or not hasattr(self,'base_actions'):return 0.
  robot_type=next(r.robot_type for r in self.runtime.robots if int(r.robot_id)==uid)
  model=self.runtime.dynamics_by_type.get(robot_type,self.runtime.dynamics)
  if type(model) is not KinematicSimple:return 0.
  key=(uid,int(task.subtask_id),tuple(uav.position),angle)
  if not hasattr(self,'preview_cache') or self.preview_tick!=self.runtime.current_step:
   self.preview_tick=self.runtime.current_step;self.preview_cache={}
  if key in self.preview_cache:return self.preview_cache[key]
  # Copy only mutable executor caches. Shared cells, graph nodes and public
  # hazard descriptors are read-only; no runtime truth positions are read.
  names=['prefixes','prefix_headings','anchors','failed','last_feedback_step','assignment_tokens','observed_positions','nearby_positions']
  def digest():return hashlib.sha256(pickle.dumps(([getattr(source,n) for n in names],source.sensing_uids,s.search_assignments,getattr(s,'_peripheral_search_targets',{})))).hexdigest()
  before=digest();rec=copy(source)
  for n in names:setattr(rec,n,dict(getattr(source,n)))
  rec.sensing_uids=set(source.sensing_uids);rec.reset_uid(uid)
  rec.observed_positions[uid]=np.asarray(uav.position,dtype=float).copy()
  shadow=copy(s);shadow._search_recovery=rec
  shadow.search_assignments={k:v for k,v in s.search_assignments.items() if int(v)!=uid}
  heat=int(task.subtask_id)-100000;shadow.search_assignments[heat]=uid
  for n in ['_peripheral_search_targets','_search_scan_bearings','_search_scan_phases','_search_scan_owners','search_dwell_by_heat']:
   setattr(shadow,n,dict(getattr(s,n,{})))
  shadow._search_viewpoints=copy(s._search_viewpoints)
  robot=SimpleNamespace(robot_id=uid,robot_type=robot_type,position=np.asarray(uav.position,dtype=float).copy(),heading=float(angle),velocity=float(uav.velocity))
  agent=s._marvel_agents.get(uid);anchor=np.asarray(task.position,dtype=float)
  base=self.base_actions[next(i for i,r in enumerate(self.runtime.robots) if int(r.robot_id)==uid)]
  if np.linalg.norm(robot.position-anchor)<=s.search_reach_radius+1e-9:
   command=shadow._search_action(robot,anchor,agent,None,base,service_scan_heading=shadow._service_scan_heading(heat,robot,anchor))
  else:
   route=observed_next_hop(agent,anchor)
   command=shadow._search_action(robot,anchor,agent,route,base,route_goal_radius=s.search_reach_radius,allow_observed_frontier=True)
  assert before==digest(),'Counterfactual mutated real executor state'
  target,desired=command;position=robot.position.copy();yaw=robot.heading
  for tick in range(1,11):
   state,_=model.step(position,target,yaw,desired,uav.velocity,self.runtime.dt)
   if not source.segment_safe(position,state['position']) or s.safety.hazards.segment_intersects_warning_buffer(position,state['position'],step=self.runtime.current_step+tick):
    self.preview_cache[key]=0.;return 0.
   position=state['position'];yaw=state['heading']
  sensor=self.runtime.sensor_by_type.get(robot_type,self.runtime.sensor)
  view=IdealSensor(dict(fov=sensor.fov,range=sensor.sensor_range,los_occlusion=True),frame=source.frame)
  cells=set(map(tuple,view._visible_cells(position,yaw,self.potential)))
  fraction=len(cells&prior)/len(prior)
  self.preview_cache[key]=fraction
  return fraction
