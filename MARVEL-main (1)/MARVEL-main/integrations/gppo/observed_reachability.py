"""Optimistic observed connectivity gate, never a path generator.

UNKNOWN cells remain passable. Diagonal-only corners are excluded only when
the actual per-tick displacement bound and five-probe radius prove they cannot
be crossed. A negative result requires an observed closed cut and separation
of the entire public detection support; failure to find a known graph path is not used.
"""
import math
import numpy as np
from scipy.ndimage import label

class ObservedReachabilityGate:
    def __init__(self,frame):
        self.frame=frame;self.signature=None;self.labels=None
        self.points={};self.minimums={}

    def prepare(self,cells,connectivity=8):
        occupied=frozenset((int(col),int(row)) for (col,row),value in cells.items()
            if value==1 and self.frame.contains_cell(int(col),int(row)))
        signature=(occupied,connectivity)
        if signature==self.signature:return
        passable=np.ones((self.frame.height_cells,self.frame.width_cells),dtype=bool)
        for col,row in occupied:passable[row,col]=False
        structure=np.ones((3,3),dtype=int) if connectivity==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
        self.labels,_=label(passable,structure)
        self.signature=signature;self.points.clear();self.minimums.clear()

    def reason(self,cells,position,goal,speed,stale_age_seconds,sensor_range,heat_radius,prepared=False):
        pose=np.asarray(position,dtype=float);target=np.asarray(goal,dtype=float)
        numbers=[speed,stale_age_seconds,sensor_range,heat_radius]
        if (pose.shape!=(2,) or target.shape!=(2,) or not np.isfinite(pose).all()
                or not np.isfinite(target).all() or not all(math.isfinite(float(x)) for x in numbers)
                or speed<=0 or min(stale_age_seconds,sensor_range,heat_radius)<0):return None
        if np.any(pose<self.frame.bounds_min) or np.any(pose>self.frame.bounds_max):return None
        if not prepared:self.prepare(cells)
        uncertainty=float(speed)*float(stale_age_seconds)
        lo=np.floor((pose-uncertainty-np.asarray(self.frame.origin))/self.frame.cell_size).astype(int)
        hi=np.floor((pose+uncertainty-np.asarray(self.frame.origin))/self.frame.cell_size).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,[self.frame.width_cells-1,self.frame.height_cells-1])
        possible=np.unique(self.labels[lo[1]:hi[1]+1,lo[0]:hi[0]+1]);possible=possible[possible>0]
        if not len(possible):return None
        minimum=math.inf
        for raw_component in possible:
            component=int(raw_component);key=(component,tuple(target))
            if key not in self.minimums:
                if component not in self.points:
                    rr,cc=np.where(self.labels==component)
                    self.points[component]=np.column_stack([cc+.5,rr+.5])*self.frame.cell_size+np.asarray(self.frame.origin)
                self.minimums[key]=float(np.linalg.norm(self.points[component]-target,axis=1).min())
            minimum=min(minimum,self.minimums[key])
        allowance=float(sensor_range)+float(heat_radius)+math.sqrt(2)*self.frame.cell_size+uncertainty
        if minimum<=allowance+1e-9:return None
        return {'reason':'observed_closed_cut_outside_public_detection_support',
            'minimum_heat_distance_m':minimum,'optimistic_detection_allowance_m':allowance,
            'unknown_cells_passable':True,'diagonal_connections_passable':self.signature[1]==8,
            'possible_component_count':len(possible)}


def observed_search_impossibility(runtime,uid,position,goal,stale_age_seconds):
    if getattr(runtime,'geometry_mode',None)!='extended' or stale_age_seconds is None:return None
    profile=runtime.config.get('_gppo_profile',{})
    if 'search_guaranteed_hidden_radius' not in profile:return None
    controller=getattr(getattr(runtime,'high_level_scheduler',None),'_search_recovery',None)
    if controller is None or not getattr(controller,'_authoritative_areal_grid',False):return None
    from .search_deadline import certified_constant_search_capability
    capability=certified_constant_search_capability(runtime,uid)
    if capability is None:return None
    speed,sensor_range=capability;step_displacement=speed*float(runtime.dt)
    if step_displacement>controller.frame.cell_size+1e-9:return None
    # At <=radius/tick, valid five-probe endpoints cannot jump a diagonal
    # corner between two OCC orthogonal cells. Other diagonals already join
    # via a passable orthogonal cell, so four-connectivity is conservative.
    connectivity=4 if step_displacement<=.2+1e-9 else 8
    gate=getattr(controller,'_observed_reachability_gate',None)
    if gate is None:
        gate=ObservedReachabilityGate(controller.frame)
        controller._observed_reachability_gate=gate
        controller._observed_reachability_step=None
    stamp=(runtime.current_step,connectivity)
    if getattr(controller,'_observed_reachability_step',None)!=stamp:
        gate.prepare(controller.cells,connectivity=connectivity)
        controller._observed_reachability_step=stamp
    return gate.reason(controller.cells,position,goal,speed,stale_age_seconds,
        sensor_range,profile['search_guaranteed_hidden_radius'],prepared=True)


def observed_motion_impossibility(runtime,uid,position,goal,stale_age_seconds):
    """Conservative observed closed-cut certificate for physical goal motion."""
    if getattr(runtime,'geometry_mode',None)!='extended' or stale_age_seconds is None:return None
    controller=getattr(getattr(runtime,'high_level_scheduler',None),'_search_recovery',None)
    if controller is None or not getattr(controller,'_authoritative_areal_grid',False):return None
    from .search_deadline import certified_constant_search_capability
    capability=certified_constant_search_capability(runtime,uid)
    if capability is None:return None
    speed,_sensor_range=capability;displacement=speed*float(runtime.dt)
    if displacement>controller.frame.cell_size+1e-9:return None
    connectivity=4 if displacement<=.2+1e-9 else 8
    gate=getattr(controller,'_observed_reachability_gate',None)
    if gate is None:
        gate=ObservedReachabilityGate(controller.frame);controller._observed_reachability_gate=gate
        controller._observed_reachability_step=None
    stamp=(runtime.current_step,connectivity)
    if getattr(controller,'_observed_reachability_step',None)!=stamp:
        gate.prepare(controller.cells,connectivity=connectivity);controller._observed_reachability_step=stamp
    result=gate.reason(controller.cells,position,goal,speed,stale_age_seconds,0.,0.,prepared=True)
    if result is not None:
        result['reason']='observed_closed_cut_outside_motion_goal_support'
        result['minimum_goal_distance_m']=result.pop('minimum_heat_distance_m')
        result['optimistic_motion_allowance_m']=result.pop('optimistic_detection_allowance_m')
    return result
