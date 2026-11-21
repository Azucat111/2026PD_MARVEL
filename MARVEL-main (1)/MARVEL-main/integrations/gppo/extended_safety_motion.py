"""Extended-only Safety motion from observed static cells and shared public hazards."""
from __future__ import annotations
import numpy as np
from .safety_layer import FrozenSafetyLayer

class ObservedMotionSafetyLayer(FrozenSafetyLayer):
    def __init__(self,runtime,*,recovery,**kwargs):
        super().__init__(runtime,**kwargs)
        self.recovery=recovery
        self._release_routes={}


    def _observed_release_action(self, robot, *, step):
        """Reuse observed Dijkstra over the existing short Safety candidates.

        Only after leaving the warning band, a legal detour may temporarily
        reduce clearance while progressing to the unchanged safe margin.
        A route prefix is retained and rechecked every tick; private map and
        hidden targets are never inputs. This bounded graph is not a global
        reachability claim.
        """
        from types import SimpleNamespace
        from utils.dynamics_models import KinematicSimple
        from .extended_travel import extended_observed_travel_time_matrix
        rec=self.recovery;uid=int(robot.robot_id);start=np.asarray(robot.position,dtype=float)
        model=self.runtime.dynamics_by_type.get(robot.robot_type,self.runtime.dynamics)
        if not isinstance(model,KinematicSimple):return None
        risk=self.hazards.risk_at(start,step=step)
        if risk is None or risk.within_warning_margin:
            self._release_routes.pop(uid,None);return None
        if self.hazards.is_safe(start,step=step):
            self._release_routes.pop(uid,None)
            return start.copy(),float(robot.heading)
        peers=rec.nearby_positions.get(uid,[])
        limits=[min(.5,float(np.linalg.norm(start-p))) for p in peers]
        failed=[old for old,origin in rec.failed.get(uid,[]) if np.linalg.norm(start-origin)<=.5]
        # Scoped to this selection: observations, peers and hazards stay fixed.
        static_cache={};clear_cache={}
        def static_clear(a,b):
            key=(tuple(np.asarray(a,dtype=float)),tuple(np.asarray(b,dtype=float)))
            if key not in static_cache:static_cache[key]=rec.segment_safe(a,b)
            return static_cache[key]
        def clear(a,b):
            key=(tuple(np.asarray(a,dtype=float)),tuple(np.asarray(b,dtype=float)))
            if key not in clear_cache:clear_cache[key]=uncached_clear(a,b)
            return clear_cache[key]
        def uncached_clear(a,b):
            a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
            if not static_clear(a,b) or self.hazards.segment_intersects_warning_buffer(a,b,step=step):return False
            d=b-a;s=float(np.dot(d,d))
            for peer,limit in zip(peers,limits):
                t=float(np.clip(np.dot(peer-a,d)/s,0.,1.)) if s>1e-20 else 0.
                if np.linalg.norm(a+t*d-peer)<limit-1e-9:return False
            return True
        def executable(point):
            if not clear(start,point) or any(np.linalg.norm(point-p)<1e-6 for p in failed):return False
            d=point-start;heading=float(np.degrees(np.arctan2(d[1],d[0]))%360.)
            update,_=model.step(start,point,robot.heading,heading,robot.velocity,self.runtime.dt)
            return clear(start,update['position']) and not self.hazards.segment_intersects_no_fly(start,update['position'],margin=.2,step=step+1)
        def cached_action():
            route=self._release_routes.get(uid)
            if not route:return None
            if not self.hazards.is_safe(route[-1],step=step):return None
            while route and np.linalg.norm(route[0]-start)<=1e-8:route.pop(0)
            if not route or not executable(route[0]):return None
            point=route[0];d=point-start
            return point.copy(),float(np.degrees(np.arctan2(d[1],d[0]))%360.)
        action=cached_action()
        if action is not None:return action
        self._release_routes.pop(uid,None)
        # Same 36 headings and 1/.5/.25/.125m lengths as existing Safety.
        points=[start+length*np.array([np.cos(angle),np.sin(angle)])
            for angle in np.linspace(0.,2*np.pi,self.num_angle_bins,endpoint=False)
            for length in (1.,.5,.25,.125)]
        points=[p for p in points if clear(p,p)]
        if not points:return None
        coords=np.asarray(points);distance=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=2)
        adjacency=np.where(distance<=1.+1e-9,0,1)
        current=int(np.argmin(np.linalg.norm(coords-start,axis=1)))
        agent=SimpleNamespace(node_coords=coords,adjacent_matrix=adjacency,current_index=current,
            neighbor_indices=np.arange(len(coords)))
        goals=sorted((p for p in points if self.hazards.is_safe(p,step=step)),
            key=lambda p:(float(np.linalg.norm(p-start)),float(p[0]),float(p[1])))[:16]
        if not goals:return None
        view=SimpleNamespace(segment_safe=static_clear,observed_positions={uid:start.copy()})
        details={}
        extended_observed_travel_time_matrix([SimpleNamespace(uav_id=uid,position=start,velocity=1.)],
            [SimpleNamespace(subtask_id=i,position=p) for i,p in enumerate(goals)],{uid:agent},
            prefix_controller=view,diagnostics=details,segment_filter=clear,include_routes=True)
        routes=[]
        for i,goal in enumerate(goals):
            row=details[i,uid]
            if not row['route_first_segment_certified'] or not row['route_goal_connected']:continue
            route=[]
            for raw in row['route_waypoints']:
                point=np.asarray(raw,dtype=float)
                if np.linalg.norm(point-(start if not route else route[-1]))>1e-8:route.append(point)
            if not route or not executable(route[0]):continue
            allpoints=[start]+route;cost=sum(float(np.linalg.norm(b-a)) for a,b in zip(allpoints,allpoints[1:]))
            routes.append((cost,i,route))
        if not routes:return None
        self._release_routes[uid]=min(routes,key=lambda r:r[:2])[2]
        return cached_action()

    def _select_certified_escape(self,robot,*,step):
        from utils.dynamics_models import KinematicSimple
        start=np.asarray(robot.position,dtype=float);uid=int(robot.robot_id)
        rec=self.recovery;agent=self._marvel_agents.get(uid)
        model=self.runtime.dynamics_by_type.get(robot.robot_type,self.runtime.dynamics)
        public=[]
        for obstacle in self.runtime.obstacles.dynamic_obstacles:
            radius=self.hazards._radius_at(obstacle,step)
            if radius is not None:
                public.append((np.asarray(obstacle.position,dtype=float),float(radius),
                    float(self.hazards._radius_at(obstacle,step+1))))
        if not public or rec is None:return start.copy(),float(robot.heading)
        release=self._observed_release_action(robot,step=step)
        if release is not None:return release
        targets=[]
        if agent is not None and agent.node_coords is not None and agent.neighbor_indices is not None:
            coords=np.asarray(agent.node_coords,dtype=float)
            targets.extend(coords[int(i),:2] for i in np.unique(agent.neighbor_indices) if 0<=int(i)<len(coords))
        risk=self.hazards.risk_at(start,step=step)
        away=start-np.asarray(risk.center,dtype=float)
        if np.linalg.norm(away)>1e-9:targets.append(start+away/np.linalg.norm(away))
        for angle in np.linspace(0.,2*np.pi,self.num_angle_bins,endpoint=False):
            targets.append(start+np.array([np.cos(angle),np.sin(angle)]))
        # Reuse the existing Safety heading bins. Circle exits are public
        # geometric candidates, then certified against observed static cells.
        for obstacle in self.runtime.obstacles.dynamic_obstacles:
            if self.hazards._radius_at(obstacle,step) is None:continue
            center=np.asarray(obstacle.position,dtype=float)
            radius=float(obstacle.max_radius)+.2+1e-6
            offset=start-center
            if np.linalg.norm(offset)>=radius:continue
            for angle in np.linspace(0.,2*np.pi,self.num_angle_bins,endpoint=False):
                ray=np.array([np.cos(angle),np.sin(angle)])
                b=float(np.dot(offset,ray))
                exit_length=-b+np.sqrt(max(0.,b*b+radius*radius-float(np.dot(offset,offset))))
                if exit_length>.1:targets.append(start+exit_length*ray)
        peers=rec.nearby_positions.get(uid,[])
        failed=[old for old,origin in rec.failed.get(uid,[]) if np.linalg.norm(start-origin)<=.5]
        fast_simple=(type(model) is KinematicSimple and
            getattr(model.step,'__func__',None) is KinematicSimple.step and
            getattr(model.position_after,'__func__',None) is KinematicSimple.position_after)
        forecast_table=[]
        if fast_simple:
            forecast_table=[(np.asarray(o.position,dtype=float),
                [self.hazards._radius_at(o,step+t) for t in range(1,81)])
                for o in self.runtime.obstacles.dynamic_obstacles]
        candidates=[]
        for order,target in enumerate(targets):
            delta=np.asarray(target,dtype=float)-start;distance=float(np.linalg.norm(delta))
            if distance<=.1:continue
            lengths=([distance] if distance<=6. else [])+[x for x in (1.,.5,.25,.125) if x<distance]
            for length in lengths:
                point=start+delta/distance*length
                if any(np.linalg.norm(point-old)<1e-6 for old in failed):continue
                if not rec.segment_safe(start,point):continue
                direction=point-start
                # A farther endpoint alone does not prove an outward escape.
                # From the real pose, straight motion must not approach any
                # nearby active shared hazard before the next physics tick.
                if any(np.linalg.norm(start-center)-radius<=self.hazards.warning_margin
                    and np.dot(direction,start-center)<-1e-9 for center,radius,_ in public):continue
                if any(np.linalg.norm(start+np.clip(np.dot(peer-start,direction)/np.dot(direction,direction),0.,1.)*direction-peer)
                    <min(.5,float(np.linalg.norm(start-peer)))-1e-9 for peer in peers):continue
                heading=float(np.degrees(np.arctan2(direction[1],direction[0]))%360.)
                update,_=model.step(start,point,robot.heading,heading,robot.velocity,self.runtime.dt)
                proposal=update['position']
                clearances=[float(np.linalg.norm(proposal-center))-future-.2 for center,_,future in public]
                if min(clearances)<=1e-9:continue
                duration=max(1,int(np.ceil(length/max(model.velocity*self.runtime.dt,1e-9))))
                for obstacle in self.runtime.obstacles.dynamic_obstacles:
                    if self.hazards._radius_at(obstacle,step) is not None and obstacle.expansion_rate>0:
                        duration=max(duration,int(np.ceil(max(0.,obstacle.max_radius-self.hazards._radius_at(obstacle,step))/obstacle.expansion_rate)))
                duration=min(80,duration)
                simulated=start.copy();yaw=float(robot.heading);forecast_clear=True;minimum_clearance=float('inf')
                for tick in range(1,duration+1):
                    if fast_simple:
                        simulated=model.position_after(simulated,point,self.runtime.dt)
                    else:
                        predicted,_=model.step(simulated,point,yaw,heading,robot.velocity,self.runtime.dt)
                        simulated=predicted['position'];yaw=predicted['heading']
                    for obstacle in self.runtime.obstacles.dynamic_obstacles:
                        future=self.hazards._radius_at(obstacle,step+tick)
                        if future is None:continue
                        clearance=float(np.linalg.norm(simulated-np.asarray(obstacle.position)))-future-.2
                        minimum_clearance=min(minimum_clearance,clearance)
                        if clearance<=1e-9:forecast_clear=False;break
                    if not forecast_clear:break
                    # Exact simple-model hold once the remaining distance is
                    # within the model's own 1e-9 hold threshold. Future public
                    # radii still certify every remaining tick via their maximum.
                    if fast_simple and np.linalg.norm(point-simulated)<=1e-9:
                        tail=[];finite=True
                        for center,radii in forecast_table:
                            remaining=[r for r in radii[tick:duration] if r is not None]
                            if remaining:
                                if not np.isfinite(remaining).all():finite=False;break
                                tail.append((center,max(remaining)))
                        if finite:
                            for center,radius in tail:
                                clearance=float(np.linalg.norm(simulated-center))-radius-.2
                                minimum_clearance=min(minimum_clearance,clearance)
                                if clearance<=1e-9:forecast_clear=False;break
                            break
                candidates.append((not forecast_clear,-minimum_clearance if forecast_clear else -min(clearances),-length,order,point,heading))
        if not candidates:
            self.escape_waypoint_fallback_count+=1
            return start.copy(),float(robot.heading)
        selected=min(candidates,key=lambda row:row[:4])
        return selected[4].copy(),selected[5]

    def select_action(self,robot,*,step):
        pending=getattr(self,'_sensing_pending',{});self._sensing_pending=pending;uid=int(robot.robot_id)
        attempted=getattr(self,'_sensing_attempted',set());self._sensing_attempted=attempted
        generation=getattr(self.recovery,'reset_generation',0)
        if generation!=getattr(self,'_sensing_recovery_generation',generation):pending.clear();attempted.clear()
        self._sensing_recovery_generation=generation
        previous=getattr(self,'_sensing_last_step',None)
        if previous is not None and step<previous:pending.clear();attempted.clear()
        self._sensing_last_step=int(step)
        token=(self.recovery.last_feedback_step.get(uid),len(self.recovery.cells));original=None
        if uid in pending:
            p=pending[uid]
            if token!=p['observation_token']:
                original=self._select_certified_escape(robot,step=step)
                if np.linalg.norm(original[0]-robot.position)>1e-9:
                    pending.pop(uid,None);return original
                p['observation_token']=token
            if p['cell'] not in self.recovery.cells and step<=p['expires'] and self._probe_clear(robot,p['target'],step):
                return p['target'].copy(),p['heading']
            pending.pop(uid,None)
        if original is None:original=self._select_certified_escape(robot,step=step)
        if np.linalg.norm(original[0]-robot.position)>1e-9:return original
        probe=self._observed_sensing_prefix(robot,step=step)
        return original if probe is None else probe
    def _probe_clear(self,robot,target,step):
        start=np.asarray(robot.position,dtype=float);rec=self.recovery
        if not rec.segment_safe(start,target):return False
        if any(np.linalg.norm(target-old)<1e-6 and np.linalg.norm(start-origin)<=.5 for old,origin in rec.failed.get(int(robot.robot_id),[])):return False
        delta=target-start;denom=float(delta@delta)
        if denom<1e-12:return False
        peers=rec.nearby_positions.get(int(robot.robot_id),[])
        if any(np.linalg.norm(start+np.clip((p-start)@delta/denom,0,1)*delta-p)<min(.5,float(np.linalg.norm(start-p)))-1e-9 for p in peers):return False
        return not any(self.hazards.segment_intersects_no_fly(start,target,margin=.2,step=step+t) for t in range(1,11))
    def _observed_sensing_prefix(self,robot,*,step):
        from utils.dynamics_models import KinematicSimple
        sensor=getattr(self.runtime,'sensor_by_type',{}).get(robot.robot_type,getattr(self.runtime,'sensor',None))
        model=self.runtime.dynamics_by_type.get(robot.robot_type,self.runtime.dynamics)
        if sensor is None or type(model) is not KinematicSimple:return None
        fov=float(sensor.fov);sensor_range=float(sensor.sensor_range);start=np.asarray(robot.position,dtype=float);rec=self.recovery
        if rec is None or not np.isfinite([fov,sensor_range,model.velocity]).all() or sensor_range<=0 or model.velocity<=0 or not 0<fov<360:return None
        from utils.sensor_models import IdealSensor
        observer=IdealSensor(dict(fov=fov,range=sensor_range,los_occlusion=True),frame=rec.frame)
        potential=np.zeros(rec.frame.shape,dtype=np.uint8)
        for (c,r),value in rec.cells.items():
            if value==1 and rec.frame.contains_cell(c,r):potential[r,c]=1
        risk=self.hazards.risk_at(start,step=step)
        if risk is None:return None
        col,row=rec.frame.world_to_cell_floor(start);look=[]
        def error(a,b):return abs((a-b+180.)%360.-180.)
        for c in range(col-1,col+2):
            for r in range(row-1,row+2):
                if not rec.frame.contains_cell(c,r) or (c,r) in rec.cells or (int(robot.robot_id),c,r) in getattr(self,'_sensing_attempted',set()):continue
                physical_center=np.asarray(rec.frame.origin)+(np.array([c,r])+.5)*rec.frame.cell_size
                if np.linalg.norm(physical_center-risk.center)<np.linalg.norm(start-risk.center)+.05:continue
                # Match IdealSensor exactly; physical cell centre is a separate quantity.
                point=np.asarray(rec.frame.cell_to_world((c,r)),dtype=float)
                if np.linalg.norm(point-start)>sensor_range:continue
                direction=point-start;bearing=float(np.degrees(np.arctan2(direction[1],direction[0]))%360.)
                if error(bearing,robot.heading)<=fov/2:continue
                look.append((c,r,point,bearing))
        candidates=[]
        for order,angle in enumerate(np.linspace(0.,2*np.pi,self.num_angle_bins,endpoint=False)):
            target=start+np.array([np.cos(angle),np.sin(angle)])
            if not self._probe_clear(robot,target,step):continue
            for c,r,point,bearing in look:
                p=start.copy();yaw=float(robot.heading);legal=True
                for tick in range(1,4):
                    state,_=model.step(p,target,yaw,bearing,robot.velocity,self.runtime.dt);p=state['position'];yaw=state['heading']
                    if self.hazards.segment_intersects_no_fly(start,p,margin=.2,step=step+tick):legal=False;break
                if not legal or np.linalg.norm(point-p)>sensor_range or not observer._has_line_of_sight(p,(c,r),potential):continue
                direction=point-p;after=float(np.degrees(np.arctan2(direction[1],direction[0]))%360.)
                angular=error(after,yaw)
                if angular>fov/2:continue
                candidates.append((angular,order,c,r,target.copy(),bearing))
        if not candidates:return None
        chosen=min(candidates,key=lambda x:x[:4]);uid=int(robot.robot_id)
        pending=getattr(self,'_sensing_pending',{});self._sensing_pending=pending
        pending[uid]=dict(cell=(chosen[2],chosen[3]),target=chosen[-2].copy(),heading=chosen[-1],expires=int(step)+10,observation_token=(rec.last_feedback_step.get(uid),len(rec.cells)))
        attempted=getattr(self,'_sensing_attempted',set());self._sensing_attempted=attempted;attempted.add((uid,chosen[2],chosen[3]))
        return chosen[-2],chosen[-1]
