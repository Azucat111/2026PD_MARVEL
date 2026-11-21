"""Diagnostic teachers reuse exact sensor geometry and existing SciPy matching."""
from copy import copy
import numpy as np
from utils.sensor_models import IdealSensor
from utils.geometry import iter_cells_in_radius
from integrations.gppo.task_graph import TaskType
from integrations.gppo.adapter import GPPOAssignment
from integrations.rule_scheduling import choose_rule_assignment
from scripts.observed_matching_assignment import matching_assignment
class PublicVisibilityTeacher:
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
    sensor=self.runtime.sensor_by_type.get(robot_types[int(uav.uav_id)],self.runtime.sensor)
    key=(int(uav.uav_id),tuple(uav.position),angle)
    if key not in self.visible:
     view=IdealSensor(dict(fov=sensor.fov,range=sensor.sensor_range,los_occlusion=True),frame=rec.frame)
     self.visible[key]=set(map(tuple,view._visible_cells(np.asarray(uav.position),angle,self.potential)))
    fractions[ti,ui]=len(self.visible[key]&prior)/len(prior)
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
