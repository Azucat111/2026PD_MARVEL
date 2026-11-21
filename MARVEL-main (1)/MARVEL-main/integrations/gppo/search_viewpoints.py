"""Extended observed-only sensing viewpoints; existing travel owns route certificates."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
from utils.geometry import iter_cells_in_radius
from utils.sensor_models import IdealSensor
from .extended_travel import extended_observed_travel_time_matrix

@dataclass(frozen=True)
class SearchViewpoint:
    position: np.ndarray
    heading: float
    potential_new_cells: int
    observed_route_length_m: float

class ObservedSearchViewpoints:
    def __init__(self, frame, *, sensor_range=10., fov=120., prior_radius=5.5, max_candidates=64):
        values=(float(sensor_range),float(fov),float(prior_radius))
        if not all(math.isfinite(v) for v in values) or values[0]<=0 or not 0<values[1]<=360 or values[2]<0:
            raise ValueError('Invalid public sensor/prior geometry')
        if int(max_candidates)<1:raise ValueError('Positive candidate bound required')
        self.frame=frame;self.sensor_range,self.fov,self.prior_radius=values
        self.max_candidates=int(max_candidates)
        self.sensor=IdealSensor({'range':self.sensor_range,'fov':self.fov,'los_occlusion':True},frame=frame)
        self.checked_cells_by_heat={}

    def observe(self, observations, assignments):
        expected=self.frame.describe()
        for heat,uid in assignments.items():
            observation=observations.get(int(uid),{})
            description=observation.get('occupancy_frame',{})
            if not all(description.get(k)==expected[k] for k in ('origin','cell_size','width_cells','height_cells')):continue
            checked=self.checked_cells_by_heat.setdefault(int(heat),set())
            for col,row,value in observation.get('visible_occupancy',[]):
                col,row,value=int(col),int(row),int(value)
                if value in (0,1) and self.frame.contains_cell(col,row):checked.add((col,row))

    def select(self, uid, heat_id, anchor, agent, recovery, segment_filter=None):
        uid=int(uid);origin=recovery.observed_positions.get(uid)
        if origin is None or agent is None:return None
        origin=np.asarray(origin,dtype=float);anchor=np.asarray(anchor,dtype=float)
        if origin.shape!=(2,) or anchor.shape!=(2,) or not np.isfinite(origin).all() or not np.isfinite(anchor).all():return None
        coords=np.asarray(getattr(agent,'node_coords',[]),dtype=float)
        if coords.ndim!=2 or coords.shape[1]!=2 or not np.isfinite(coords).all():return None
        # UNKNOWN is transparent only for potential information gain. Every
        # flight route is still certified by the original observed FREE gate.
        potential=np.zeros(self.frame.shape,dtype=np.uint8)
        for (col,row),value in recovery.cells.items():
            if self.frame.contains_cell(col,row) and value==1:potential[row,col]=1
        prior=set((int(col),int(row)) for col,row in iter_cells_in_radius(self.frame,anchor,self.prior_radius)
            if recovery.cells.get((int(col),int(row)))!=1
            and np.linalg.norm(self.frame.cell_to_world((int(col),int(row)))-anchor)<=self.prior_radius)
        checked=self.checked_cells_by_heat.get(int(heat_id),set());candidates=[]
        ordered=sorted(range(len(coords)),key=lambda i:(float(np.linalg.norm(coords[i]-anchor)),*coords[i],i))
        for i in ordered:
            point=coords[i]
            if np.linalg.norm(point-anchor)>self.sensor_range+self.prior_radius:continue
            if np.linalg.norm(point-origin)<=.1:continue # No stationary yaw assumption.
            if not recovery.segment_safe(point,point) or (segment_filter is not None and not segment_filter(point,point)):continue
            heading=float(np.degrees(np.arctan2(*(anchor-point)[::-1]))%360.)
            visible=set(map(tuple,self.sensor._visible_cells(point,heading,potential)))
            gain=len((visible&prior)-checked)
            if gain:candidates.append((point.copy(),heading,gain))
            if len(candidates)>=self.max_candidates:break
        if not candidates:return None
        details={}
        tasks=[SimpleNamespace(subtask_id=i,position=point) for i,(point,_,_) in enumerate(candidates)]
        extended_observed_travel_time_matrix([SimpleNamespace(uav_id=uid,position=origin,velocity=1.)],tasks,{uid:agent},
            prefix_controller=recovery,diagnostics=details,segment_filter=segment_filter,include_routes=True)
        ranked=[]
        for i,(point,heading,gain) in enumerate(candidates):
            route=details[(i,uid)]
            if not route['route_goal_connected'] or not route['route_first_segment_certified']:continue
            points=[origin]+[np.asarray(p,dtype=float) for p in route['route_waypoints']]
            distance=sum(float(np.linalg.norm(b-a)) for a,b in zip(points,points[1:]))
            ranked.append((-gain/max(1.,distance),float(point[0]),float(point[1]),SearchViewpoint(point,heading,gain,distance)))
        return None if not ranked else min(ranked,key=lambda r:r[:3])[3]
