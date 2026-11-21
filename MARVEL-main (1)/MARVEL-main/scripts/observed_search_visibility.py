"""Public prior support/current own-camera overlap. No hidden-target or map inputs.
This is an information feasibility probe, not a learned GPPO actor or oracle.
"""
import math
import numpy as np
from integrations.gppo.task_graph import TaskType
from integrations.gppo.adapter import GPPOAssignment

class ObservedSearchVisibility:
    def __init__(self,frame,support_radius,age_fraction=.05):
        if frame.is_native:raise ValueError('Extended experimental observations only')
        if not math.isfinite(support_radius) or support_radius<0:raise ValueError('Public finite support radius required')
        if not math.isfinite(age_fraction) or not 0<=age_fraction<=1:raise ValueError("Valid shared age fraction required")
        self.age_fraction=float(age_fraction)
        self.frame=frame;self.radius=float(support_radius);self.occupied=set();self.current={};self.previous={};self.cache={}

    def observe(self,observations):
        self.previous=self.current.copy();self.current={};self.cache={}
        expected=self.frame.describe()
        for uid,obs in observations.items():
            if obs.get('visible_occupancy_semantics')!='areal_file_grid' or obs.get('occupancy_frame')!=expected:continue
            raw=np.asarray(obs.get('visible_occupancy',[]),dtype=float)
            if raw.size==0:self.current[int(uid)]=set();continue
            if raw.ndim!=2 or raw.shape[1]!=3 or not np.isfinite(raw).all() or not np.equal(raw,np.round(raw)).all():continue
            if any(not self.frame.contains_cell(int(c),int(r)) or int(label) not in (0,1) for c,r,label in raw):continue
            visible=set()
            for c,r,label in raw.astype(int):
                cell=(int(c),int(r))
                if label==1:self.occupied.add(cell)
                else:self.occupied.discard(cell);visible.add(cell)
            self.current[int(uid)]=visible

    def support(self,goal):
        goal=np.asarray(goal,dtype=float)
        if goal.shape!=(2,) or not np.isfinite(goal).all():return set()
        key=tuple(goal)
        if key not in self.cache:
            x,y=goal;f=self.frame;r=self.radius;s=f.cell_size
            cells={(c,row) for c in range(max(0,math.ceil((x-r-f.origin_x)/s)),min(f.width_cells-1,math.floor((x+r-f.origin_x)/s))+1)
                for row in range(max(0,math.ceil((y-r-f.origin_y)/s)),min(f.height_cells-1,math.floor((y+r-f.origin_y)/s))+1)
                if np.linalg.norm(np.array([f.origin_x+c*s,f.origin_y+row*s])-goal)<=r+1e-9}
            # Generator uses lattice corners, not physical-cell centers.
            self.cache[key]=cells-self.occupied
        return self.cache[key]

    def features(self,goal,uid):
        if int(uid) not in self.current:return (0.,0.)
        possible=self.support(goal)
        if not possible:return (0.,0.)
        current=len(possible & self.current[int(uid)])/len(possible)
        previous=len(possible & self.previous.get(int(uid),self.current[int(uid)]))/len(possible)
        return ((1-self.age_fraction)*current+self.age_fraction*previous,1.)

def visibility_assignment(graph,tracker):
    candidates=[]
    for t,task in enumerate(graph.subtasks):
        for u,uav in enumerate(graph.uav_states):
            if graph.action_mask[t,u]:continue
            coverage=tracker.features(task.position,uav.uav_id)[0] if task.task_type==TaskType.TARGET_SEARCH and task.position is not None else 0.
            candidates.append((-float(task.priority),-coverage,float(graph.edge_features[t,u,2]),int(task.subtask_id),int(uav.uav_id),t,u))
    if not candidates:return None
    *_,t,u=min(candidates)
    return GPPOAssignment(t*graph.num_uavs+u,t,u,int(graph.subtasks[t].subtask_id),int(graph.uav_states[u].uav_id),0.)
