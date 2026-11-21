"""Conservative extended-only public-geometric true-detection lower bound.

A failed bound certifies time infeasibility; a passing bound does NOT certify a
physical path, visibility, or task completion. No hidden target/map input.
"""
import math
import numpy as np
from .task_graph import TaskType

def apply_search_deadline_bound(graph,remaining_seconds,sensor_range,heat_radius,cell_size,stale_age_seconds):
    parameters=[remaining_seconds,sensor_range,heat_radius,cell_size,stale_age_seconds]
    if not all(math.isfinite(float(x)) for x in parameters):return []
    if min(sensor_range,heat_radius,cell_size,stale_age_seconds)<0:return []
    remaining=max(0.,float(remaining_seconds));reasons=[]
    for ti,task in enumerate(graph.subtasks):
        if task.task_type!=TaskType.TARGET_SEARCH or task.position is None:continue
        goal=np.asarray(task.position,dtype=float)[:2]
        if not np.isfinite(goal).all():continue
        for ui,uav in enumerate(graph.uav_states):
            if graph.action_mask[ti,ui]:continue
            speed=float(uav.velocity);pose=np.asarray(uav.position,dtype=float)[:2]
            if not math.isfinite(speed) or speed<=0 or not np.isfinite(pose).all():continue
            distance=float(np.linalg.norm(pose-goal))
            reach=float(sensor_range)+float(heat_radius)+math.sqrt(2)*float(cell_size)+speed*float(stale_age_seconds)
            lower=max(0.,distance-reach)/speed
            if lower>remaining+1e-9:
                graph.action_mask[ti,ui]=True
                reasons.append({'task_id':int(task.subtask_id),'uav_id':int(uav.uav_id),
                    'lower_bound_seconds':lower,'remaining_seconds':remaining,
                    'public_distance_m':distance,'optimistic_reach_allowance_m':reach})
    return reasons


def certified_constant_search_capability(runtime,uid):
    """Public execution/sensor capability, not an instantaneous speed guess."""
    from utils.dynamics_models import KinematicSimple
    from utils.sensor_models import IdealSensor
    robot=next((r for r in getattr(runtime,'robots',()) if int(r.robot_id)==int(uid)),None)
    if robot is None:return None
    dynamics=getattr(runtime,'dynamics_by_type',{}).get(robot.robot_type,getattr(runtime,'dynamics',None))
    sensor=getattr(runtime,'sensor_by_type',{}).get(robot.robot_type)
    if type(dynamics) is not KinematicSimple or type(sensor) is not IdealSensor:return None
    speed=float(dynamics.velocity);sensor_range=float(sensor.sensor_range)
    if not math.isfinite(speed) or speed<=0 or not math.isfinite(sensor_range) or sensor_range<0:return None
    return speed,sensor_range
