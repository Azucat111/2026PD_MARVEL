"""Read-only diagnosis output; never supplies data to actor or controller."""
import json
import numpy as np

OBSERVATION_FIELDS=('position','heading','visible_occupancy','occupancy_frame',
    'visible_occupancy_semantics','nearby_robot_positions','motion_feedback')

def json_value(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,set):return sorted(value)
    raise TypeError(type(value).__name__)

def capture_motion_frame(runtime,scheduler,observations=None,actions=None):
    active=set(map(int,scheduler.search_assignments.values()))
    active.update(int(uid) for uid,_ in scheduler.relay_assignments.values())
    active.update(map(int,getattr(getattr(scheduler,"safety",None),"active_uav_ids",set())))
    recovery=getattr(scheduler,'_search_recovery',None)
    frame=dict(step=int(runtime.current_step),time=float(runtime.current_step*runtime.dt),
        physical_poses_diagnostic_only={int(r.robot_id):dict(position=r.position,
            heading=float(r.heading),travel_distance=float(r.travel_distance)) for r in runtime.robots},
        observed_graphs_diagnostic_only={uid:dict(node_coords=getattr(agent,"node_coords",None),
            neighbor_indices=getattr(agent,"neighbor_indices",None),current_index=getattr(agent,"current_index",None),
            zero_adjacency_edges=None if getattr(agent,"adjacent_matrix",None) is None else
                np.argwhere(np.asarray(agent.adjacent_matrix)==0))
            for uid,agent in getattr(scheduler,"_marvel_agents",{}).items() if int(uid) in active},
        public_safety_state=dict(active=sorted(map(int,getattr(getattr(scheduler,"safety",None),"active_uav_ids",set()))),
            safe_counters=dict(getattr(getattr(scheduler,"safety",None),"safe_counter",{})),
            cached_release_routes=dict(getattr(getattr(scheduler,"safety",None),"_release_routes",{}))),
        public_search_assignments=dict(scheduler.search_assignments),
        observed_search_viewpoint_state=dict(
            cached_targets=dict(getattr(scheduler,'_peripheral_search_targets',{})),
            checked_cells_by_heat=dict(getattr(getattr(scheduler,'_search_viewpoints',None),'checked_cells_by_heat',{}))),
        public_search_service_state=dict(dwell=dict(getattr(scheduler,"search_dwell_by_heat",{})),
            scan_bearings=dict(getattr(scheduler,"_search_scan_bearings",{})),
            scan_phases=dict(getattr(scheduler,"_search_scan_phases",{})),
            scan_owners=dict(getattr(scheduler,"_search_scan_owners",{})),
            serviced=sorted(getattr(scheduler,"serviced_heat_ids",set()))),
        public_heat_positions={int(uid):point.position for uid,point in scheduler.heat_by_id.items()},
        public_relay_assignments={int(target):dict(helper=int(uid),anchor=anchor)
            for target,(uid,anchor) in scheduler.relay_assignments.items()},
        own_motion_feedback=dict(getattr(runtime,'_motion_feedback',{})),
        observed_recovery=None if recovery is None else dict(frame=recovery.frame.describe(),
            cells=[[int(c),int(r),int(label)] for (c,r),label in sorted(recovery.cells.items())],
            prefixes=dict(recovery.prefixes),prefix_headings=dict(recovery.prefix_headings),
            anchors=dict(recovery.anchors),failed=dict(recovery.failed)),
        selected_own_observations={} if observations is None else
            {uid:{key:observations[uid][key] for key in OBSERVATION_FIELDS if key in observations[uid]}
             for uid in sorted(active) if uid in observations})
    if actions is not None:
        frame['actual_commands']={int(robot.robot_id):dict(waypoint=action[0],heading=action[1])
            for robot,action in zip(runtime.robots,actions) if action is not None}
    return frame

def append_motion_frame(file,frame):
    with file.open('a') as handle:handle.write(json.dumps(frame,default=json_value)+'\n')
