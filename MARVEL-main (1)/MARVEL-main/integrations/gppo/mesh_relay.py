"""Extended Relay semantics on the physical UAV mesh, using public positions only.
Radio forecasts are estimates, not a motion planner or physical-path certificate.
Native uses the unchanged frozen base-rooted builder/release classes.
"""
from __future__ import annotations
import numpy as np
from .event_sources import (ObservedRelayDemandBuilder,RelayObservedSnapshot,RelayDemand,
    RelayDemandResult,FROZEN_RELAY_PROCESSING_TIME)
from .event_allocator import GPPOEventSlot
from .task_graph import TaskType
from .relay_release import FrozenRelayReleaseGate

def extended_relay_helper_budget(fleet_size,frozen_budget):
    """Bound supplements after actual arrival; initial and native budgets stay frozen."""
    return max(int(frozen_budget),(int(fleet_size)+9)//10)

def mesh_components(positions,comm_range,exclude_uav_ids=()):
    excluded=set(map(int,exclude_uav_ids))
    points={int(uid):np.asarray(p,dtype=float) for uid,p in positions.items() if int(uid) not in excluded}
    if any(p.shape!=(2,) or not np.isfinite(p).all() for p in points.values()):
        raise ValueError('Finite public XY positions required')
    unseen=set(points);groups=[]
    while unseen:
        start=min(unseen);unseen.remove(start);component={start};pending=[start]
        while pending:
            uid=pending.pop()
            neighbors=sorted(v for v in unseen if np.linalg.norm(points[uid]-points[v])<=comm_range)
            for v in neighbors:unseen.remove(v);component.add(v);pending.append(v)
        groups.append(tuple(sorted(component)))
    return sorted(groups,key=lambda group:(-len(group),group))

class ObservedMeshRelayDemandBuilder(ObservedRelayDemandBuilder):
    def snapshot(self,positions,*,base_position=None,exclude_uav_ids=()):
        groups=mesh_components(positions,self.comm_range,exclude_uav_ids)
        active=tuple(sorted(uid for group in groups for uid in group))
        connected=groups[0] if groups else ()
        connected_set=set(connected);isolated=tuple(uid for uid in active if uid not in connected_set)
        return RelayObservedSnapshot(len(connected)/len(active) if active else 1.,connected,isolated,active,
            {uid:0 if uid in connected_set else -1 for uid in active})

    def _choose_anchor(self,group,backbone,positions,old_anchor=None,anchor_scorer=None):
        distance,target,anchor_uid=min((float(np.linalg.norm(np.asarray(positions[t])-positions[a])),t,a)
            for t in group for a in backbone)
        provider=getattr(self,'anchor_candidates_provider',None)
        if provider is None:
            if distance>2*self.comm_range:return None
            return distance,target,anchor_uid,.5*(np.asarray(positions[target])+positions[anchor_uid])
        points=np.asarray(provider(),dtype=float)
        if points.size==0:return None
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Observed safe XY anchor candidates required')
        targets=np.asarray([positions[uid] for uid in group]);cores=np.asarray([positions[uid] for uid in backbone])
        target_dist=np.linalg.norm(points[:,None,:]-targets[None,:,:],axis=2)
        core_dist=np.linalg.norm(points[:,None,:]-cores[None,:,:],axis=2)
        ti=target_dist.argmin(axis=1);ai=core_dist.argmin(axis=1);rows=np.arange(len(points))
        valid=np.flatnonzero((target_dist[rows,ti]<=self.comm_range)&(core_dist[rows,ai]<=self.comm_range))
        scores=np.zeros(len(points)) if anchor_scorer is None else np.asarray(anchor_scorer(points),dtype=float)
        if scores.shape!=(len(points),) or not np.isfinite(scores).all():raise ValueError('Finite candidate radio scores required')
        choices=[]
        for index in valid:
            target=int(group[ti[index]]);anchor_uid=int(backbone[ai[index]])
            gap=float(np.linalg.norm(targets[ti[index]]-cores[ai[index]]))
            midpoint=.5*(targets[ti[index]]+cores[ai[index]])
            old=old_anchor is not None and np.linalg.norm(points[index]-old_anchor)<=1e-9
            choices.append((not old,float(np.linalg.norm(points[index]-midpoint)),gap,target,anchor_uid,
                float(points[index,0]),float(points[index,1]),points[index],float(scores[index])))
        if not choices:return None
        choice=min(choices,key=lambda row:(-row[8],*row[:7]))
        return choice[2],choice[3],choice[4],choice[7].copy()

    def build(self,positions,*,base_position=None,source_task_id='T3_relay',snapshot=None):
        snap=snapshot or self.snapshot(positions)
        groups=mesh_components(positions,self.comm_range)
        if len(groups)<2:return RelayDemandResult(snap,(),())
        backbone=groups[0];candidates=[]
        for group in groups[1:]:
            solution=self._choose_anchor(group,backbone,positions)
            if solution is None:continue
            distance,target,anchor_uid,midpoint=solution
            # base_distance is the retained record field; here it means boundary gap.
            demand=RelayDemand(target,tuple(map(float,positions[target])),tuple(map(float,midpoint)),
                True,distance,snap.hop_counts[target])
            candidates.append((-len(group),distance,target,anchor_uid,demand))
        candidates.sort(key=lambda row:row[:4]);selected=candidates[:self.max_demands]
        endpoints={uid for row in selected for uid in row[2:4]}
        slots=tuple(GPPOEventSlot(slot_id=200000+row[2],task_type=TaskType.RELAY,priority=2.,
            position=row[4].anchor_position,processing_time=FROZEN_RELAY_PROCESSING_TIME,
            source_task_id=source_task_id,source_entity_id=row[2],forbidden_uav_ids=tuple(sorted(endpoints)))
            for row in selected)
        return RelayDemandResult(snap,tuple(row[4] for row in selected),slots)

    def _candidate_lcc_sizes(self,points,positions,helper,assignments):
        """Vectorized radio forecasts for existing observed anchor candidates."""
        fixed={int(uid):np.asarray(p,dtype=float) for uid,p in positions.items() if int(uid)!=int(helper)}
        for other,anchor in assignments.values():
            if int(other)!=int(helper):fixed[int(other)]=np.asarray(anchor,dtype=float)
        groups=mesh_components(fixed,self.comm_range)
        touched=np.ones(len(points),dtype=float)  # The moving helper itself.
        for group in groups:
            coords=np.asarray([fixed[uid] for uid in group])
            reaches=(np.linalg.norm(np.asarray(points)[:,None,:]-coords[None,:,:],axis=2)<=self.comm_range).any(axis=1)
            touched+=len(group)*reaches
        return np.maximum(touched,max(map(len,groups),default=0))

    def refresh_assignments(self,assignments,positions,*,fixed_targets=()):
        """Jointly retarget existing bridges using only public radio positions.

        Helpers are excluded when finding disconnected component boundaries.
        A target component may itself be largest. All proposed anchors must
        preserve existing nonhelper connections and never reduce largest
        component size. Connected targets retain release-window semantics.
        This is a radio forecast; routing and physics still validate motion.
        """
        helpers={int(uid) for uid,_ in assignments.values()}
        groups=mesh_components(positions,self.comm_range,helpers)
        if len(groups)<2:return
        proposed={}
        for target,(helper,old_anchor) in tuple(assignments.items()):
            if target in fixed_targets:continue
            group=next((g for g in groups if int(target) in g),None)
            if group is None or int(helper) not in positions:continue
            others=[g for g in groups if g!=group]
            # Preserve deterministic component priority and nearest boundary.
            planned_so_far={**assignments,**proposed}
            scorer=lambda points: self._candidate_lcc_sizes(points,positions,helper,planned_so_far)
            solutions=[]
            backbones=sorted(others,key=lambda g:(-len(g),g))
            if getattr(self,"anchor_candidates_provider",None) is None:backbones=backbones[:1]
            for backbone in backbones:
                candidate=self._choose_anchor(group,backbone,positions,old_anchor,anchor_scorer=scorer)
                if candidate is None:continue
                score=float(scorer(np.asarray([candidate[3]]))[0])
                changed=not np.allclose(candidate[3],old_anchor,rtol=0,atol=1e-9)
                solutions.append((-score,changed,-len(backbone),backbone,candidate))
            if not solutions:continue
            solution=min(solutions,key=lambda row:row[:4])[4]
            gap,target_boundary,core_boundary,anchor=solution
            if not np.allclose(anchor,old_anchor,rtol=0,atol=1e-9):
                proposed[target]=(helper,anchor)
        if not proposed:return
        planned={**assignments,**proposed}
        moved={int(uid) for uid,_ in planned.values()}
        forecast=dict(positions)
        for helper,anchor in planned.values():forecast[int(helper)]=anchor
        original=mesh_components(positions,self.comm_range)
        future=mesh_components(forecast,self.comm_range)
        preserved=all(any(set(group)-moved <= set(new_group) for new_group in future)
                      for group in original)
        if preserved and max(map(len,future),default=0)>=max(map(len,original),default=0):
            assignments.update(proposed)

    def candidate_mask(self,graph,assignments):
        """Only additional hard masks; evaluate joint planned radio positions.

        All inputs are delayed public graph positions/task anchors. No map reads.
        Real movement remains subject to observed routing, physics and Shield.
        """
        positions={int(u.uav_id):np.asarray(u.position,dtype=float).copy() for u in graph.uav_states}
        anchors={int(t.subtask_id):t.position for t in graph.subtasks}
        for assigned in assignments:
            positions[int(assigned.uav_id)]=np.asarray(anchors[int(assigned.slot_id)],dtype=float)
        original_groups=mesh_components(positions,self.comm_range)
        baseline=max(map(len,original_groups),default=0)
        mask=np.zeros_like(graph.action_mask,dtype=bool)
        for ti,task in enumerate(graph.subtasks):
            if task.position is None:mask[ti,:]=True;continue
            for ui,uav in enumerate(graph.uav_states):
                if graph.action_mask[ti,ui]:continue
                motion_guard=getattr(self,"motion_candidate_guard",None)
                if motion_guard is not None and motion_guard(int(uav.uav_id),uav.position,task.position):
                    mask[ti,ui]=True;continue
                hypothetical={**positions,int(uav.uav_id):np.asarray(task.position,dtype=float)}
                forecast_groups=mesh_components(hypothetical,self.comm_range)
                count=max(map(len,forecast_groups),default=0)
                donor=int(uav.uav_id)
                preserved=all(any(set(group)-{donor} <= set(new_group) for new_group in forecast_groups)
                    for group in original_groups)
                if count<=baseline or not preserved:mask[ti,ui]=True
        return mask

class ObservedMeshRelayReleaseGate(FrozenRelayReleaseGate):
    def __init__(self,*,min_connectivity,fleet_uav_ids,**kwargs):
        super().__init__(**kwargs)
        self.min_connectivity=float(min_connectivity);self.fleet_uav_ids=frozenset(map(int,fleet_uav_ids))

    def observe(self,*,assignments,positions,relay_builder,base_position):
        # Missing public fleet state cannot certify a release.
        if set(positions)!=self.fleet_uav_ids:
            self.reset_stability();return False
        helpers={int(uid) for uid,_ in assignments.values()}
        remaining=relay_builder.snapshot(positions,exclude_uav_ids=helpers)
        self.last_counterfactual_connectivity=remaining.connectivity_ratio
        # Even if every released helper becomes isolated, full-fleet ratio must
        # still meet the task's physical minimum. Preserve .95 remaining-fleet
        # hysteresis and target membership checks from the existing gate.
        conservative=len(remaining.connected_uav_ids)/max(1,len(self.fleet_uav_ids))
        if conservative<self.min_connectivity:
            self.reset_stability();return False
        return super().observe(assignments=assignments,positions=positions,
            relay_builder=relay_builder,base_position=base_position)
