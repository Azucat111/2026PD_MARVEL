"""Extended observed travel estimate using real-pose graph attachment.

Known FREE graph paths lead to the reachable node closest to the public goal.
The final Euclidean residual through UNKNOWN remains a lower-bound assumption,
not a completion/feasibility guarantee. No truth map or hidden targets are read.
"""
import heapq
import numpy as np
from .observed_graph import _graph_state

def extended_observed_travel_time_matrix(uav_states,subtasks,agent_by_uid,
        prefix_controller=None,diagnostics=None,segment_filter=None,include_routes=False,goal_radius=0.):
    goal_radius=float(goal_radius)
    if not np.isfinite(goal_radius) or goal_radius<0:raise ValueError("Finite nonnegative goal radius required")
    clearance_cache={}
    def clear(start,end):
        key=tuple(sorted((tuple(np.asarray(start)[:2]),tuple(np.asarray(end)[:2]))))
        if key not in clearance_cache:
            clearance_cache[key]=((prefix_controller is None or prefix_controller.segment_safe(start,end))
                and (segment_filter is None or segment_filter(start,end)))
        return clearance_cache[key]
    result=np.zeros((len(subtasks),len(uav_states)),dtype=np.float32)
    for ui,uav in enumerate(uav_states):
        uid=int(uav.uav_id)
        position=np.asarray(uav.position,dtype=float)[:2]
        speed=max(float(uav.velocity),1e-6)
        agent=agent_by_uid.get(uid)
        state=_graph_state(agent)
        coords=distance=first_attachment=None
        sources=[]
        if state is not None:
            coords,adj,start=state
            if prefix_controller is None:
                # Without measured clearance, do not invent an offset link.
                if np.linalg.norm(position-coords[start,:2])<1e-9:
                    sources=[(0.,start)]
            else:
                indices=getattr(agent,'neighbor_indices',None)
                if indices is None:indices=np.flatnonzero(adj[start]==0)
                for index in np.unique(np.append(indices,start)):
                    index=int(index)
                    if not 0<=index<len(coords):continue
                    length=float(np.linalg.norm(position-coords[index,:2]))
                    if length<=6. and clear(position,coords[index,:2]):
                        sources.append((length,index))
            distance=np.full(len(coords),np.inf)
            first_attachment=np.full(len(coords),-1,dtype=int)
            next_waypoint=np.full(len(coords),-1,dtype=int) if include_routes else None
            predecessors=np.full(len(coords),-1,dtype=int) if include_routes else None
            heap=[]
            for cost,index in sources:
                distance[index]=min(distance[index],cost)
                first_attachment[index]=index
                if include_routes:next_waypoint[index]=index if cost>.1 else -1
                heapq.heappush(heap,(cost,index))
            while heap:
                cost,index=heapq.heappop(heap)
                if cost!=distance[index]:continue
                for other in np.flatnonzero(adj[index]==0):
                    other=int(other)
                    if other==index:continue
                    if not clear(coords[index,:2],coords[other,:2]):
                        continue
                    candidate=cost+float(np.linalg.norm(coords[index,:2]-coords[other,:2]))
                    if candidate<distance[other]:
                        distance[other]=candidate
                        first_attachment[other]=first_attachment[index]
                        if include_routes:
                            next_waypoint[other]=next_waypoint[index] if next_waypoint[index]>=0 else other
                            predecessors[other]=index
                        heapq.heappush(heap,(candidate,other))
        for ti,task in enumerate(subtasks):
            if task.position is None:continue
            goal=np.asarray(task.position,dtype=float)[:2]
            endpoint=None;known=0.
            residual=float(np.linalg.norm(position-goal))
            status='no_certified_attachment_euclidean_lower_bound'
            if distance is not None and np.isfinite(distance).any():
                reachable=np.flatnonzero(np.isfinite(distance))
                differences=np.linalg.norm(coords[reachable,:2]-goal,axis=1)
                endpoint=int(reachable[int(np.argmin(differences))])
                known=float(distance[endpoint])
                residual=float(np.linalg.norm(coords[endpoint,:2]-goal))
                status='known_route_plus_unknown_residual'
            result[ti,ui]=(known+residual)/speed
            if diagnostics is not None:
                # Model features remain delayed. Issue a deployment certificate
                # only for the chosen path's first attachment from own current
                # sensor position, never from runtime truth or another source.
                actual=getattr(prefix_controller,'observed_positions',{}).get(uid)
                certified=False
                if actual is not None and endpoint is not None:
                    actual=np.asarray(actual,dtype=float)
                    source=int(first_attachment[endpoint])
                    certified=(actual.shape==(2,) and np.all(np.isfinite(actual))
                        and source>=0 and np.linalg.norm(actual-coords[source,:2])<=6.
                        and clear(actual,coords[source,:2]))
                diagnostics[(int(task.subtask_id),uid)]={
                    'status':status,'source_position':position.tolist(),
                    'real_pose_clearance_certified':bool(certified),
                    'observed_destination':None if endpoint is None else coords[endpoint,:2].tolist(),
                    'known_path_m':known,'unknown_residual_lower_bound_m':residual,
                    'future_traffic_not_predicted':True}
                if include_routes:
                    # Cost lower bounds may contain UNKNOWN. Execution requires
                    # a fully observed connector from the graph to the goal.
                    route_endpoint=None;connector_end=None
                    if distance is not None:
                        connections=[]
                        for node in np.flatnonzero(np.isfinite(distance)):
                            tail=float(np.linalg.norm(coords[node,:2]-goal))
                            approach=(coords[node,:2].copy() if tail<=goal_radius else
                                goal+(coords[node,:2]-goal)*goal_radius/tail)
                            connector=float(np.linalg.norm(coords[node,:2]-approach))
                            if connector<=6. and clear(coords[node,:2],approach):
                                connections.append((float(distance[node])+connector,int(node),approach))
                        if connections:
                            _,route_endpoint,connector_end=min(connections,key=lambda row:row[:2])
                    index=-1 if route_endpoint is None else int(next_waypoint[route_endpoint])
                    point=None if index<0 else coords[index,:2]
                    if point is None and connector_end is not None and actual is not None and np.linalg.norm(connector_end-actual)>.1:
                        point=connector_end;index=int(route_endpoint)
                    source=-1 if route_endpoint is None else int(first_attachment[route_endpoint])
                    route_certified=(actual is not None and source>=0
                        and np.allclose(actual,position,rtol=0,atol=1e-9)
                        and np.linalg.norm(actual-coords[source,:2])<=6.
                        and clear(actual,coords[source,:2]))
                    executable=(route_certified and point is not None
                        and np.linalg.norm(point-actual)>.1 and clear(actual,point))
                    path=[];node=-1 if route_endpoint is None else int(route_endpoint)
                    while node>=0:
                        path.append(coords[node,:2].tolist());node=int(predecessors[node])
                    diagnostics[(int(task.subtask_id),uid)].update(
                        route_goal_connected=route_endpoint is not None,
                        route_goal_radius_m=goal_radius,
                        route_goal_connector=None if route_endpoint is None else [coords[route_endpoint,:2].tolist(),connector_end.tolist()],
                        route_next_index=index,route_next_waypoint=None if point is None else point.tolist(),
                        route_first_segment_certified=bool(executable),route_waypoints=list(reversed(path))+([] if connector_end is None else [connector_end.tolist()]))
    return result


def extended_observed_next_hop(uid,position,goal,agent,prefix_controller,segment_filter=None,route_details=None,goal_radius=0.):
    """Execute a first leg of the existing certified observed travel Dijkstra.

    Fresh own-position equality and full FREE/public-filter clearance are
    mandatory. The residual remains a lower bound, never a path certificate.
    """
    from types import SimpleNamespace
    actual=getattr(prefix_controller,'observed_positions',{}).get(int(uid))
    if actual is None or not np.allclose(actual,position,rtol=0,atol=1e-9):return None
    diagnostics={}
    extended_observed_travel_time_matrix([SimpleNamespace(uav_id=int(uid),position=actual,velocity=1.)],
        [SimpleNamespace(subtask_id=-1,position=goal)],{int(uid):agent},prefix_controller,diagnostics,segment_filter,include_routes=True,goal_radius=goal_radius)
    result=diagnostics[-1,int(uid)]
    if route_details is not None:route_details.update(result)
    if not result['route_first_segment_certified']:return None
    return np.asarray(result['route_next_waypoint'],dtype=float),int(result['route_next_index'])


# Only the observed prefix is certified; the original UNKNOWN tail remains unexecuted.
def observed_frontier_next_hop(uid,position,goal,agent,recovery,goal_details,segment_filter=None):
    position=np.asarray(position,dtype=float)[:2];goal=np.asarray(goal,dtype=float)[:2]
    actual=recovery.observed_positions.get(int(uid))
    if actual is None or not np.allclose(actual,position,rtol=0,atol=1e-9):return None
    if goal_details.get('route_goal_connected') or not goal_details.get('real_pose_clearance_certified'):return None
    endpoint=goal_details.get('observed_destination')
    if endpoint is None:return None
    endpoint=np.asarray(endpoint,dtype=float)[:2]
    if not np.all(np.isfinite(endpoint)):return None
    if np.linalg.norm(endpoint-goal)>=np.linalg.norm(position-goal)-1e-9:return None
    details={}
    hop=extended_observed_next_hop(uid,position,endpoint,agent,recovery,segment_filter=segment_filter,route_details=details,goal_radius=0.)
    if hop is None:return None
    details.update(original_search_goal_connected=False,observed_prefix_only=True,original_goal_position=goal.tolist(),unknown_tail_executed=False)
    return hop,details
