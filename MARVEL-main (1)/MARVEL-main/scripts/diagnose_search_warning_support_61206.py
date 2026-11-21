"""Read-only public-warning vs captured Search service support diagnosis."""
from pathlib import Path
import json,sys,types,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.extended_travel import extended_observed_next_hop
PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt8/episode_010/gppo'
OUT=ROOT/'artifacts/extended_dynamic_search_cycle_61206_20261010_attempt1'
frames=json.loads((OUT/'captured_frames.json').read_text());f=next(x for x in frames if x['step']==400);uid=28;goal=np.asarray(f['public_heat_positions']['2'],dtype=float)
cfg=json.loads((PAIR/'config.json').read_text());meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
own=f['selected_own_observations'][str(uid)];origin=np.asarray(own['position'])
om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);om.step(f['step'],.1)
runtime=types.SimpleNamespace(current_step=f['step'],obstacles=om,geometry_mode='extended');haz=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
checks=[]
for warning in (False,True):
 details={};filter_segment=(lambda a,b:not haz.segment_intersects_warning_buffer(a,b,step=f['step'])) if warning else None
 hop=extended_observed_next_hop(uid,origin,goal,agent,rec,segment_filter=filter_segment,route_details=details,goal_radius=3.5)
 checks.append(dict(public_warning_enforced=warning,next_hop=None if hop is None else hop[0].tolist(),route_goal_connected=details.get('route_goal_connected'),route_waypoints=details.get('route_waypoints')))
geometry=[]
for obstacle in om.dynamic_obstacles:
 radius=haz._radius_at(obstacle,f['step']);center=np.asarray(obstacle.position);distance=float(np.linalg.norm(goal-center));warning_radius=radius+haz.warning_margin
 geometry.append(dict(center=center.tolist(),radius_m=radius,warning_radius_m=warning_radius,heat_center_distance_m=distance,service_radius_m=3.5,entire_service_disk_excluded=distance+3.5<=warning_radius,range10_guaranteed_hidden5_5_support_radius_m=4.5,entire_guaranteed_range_support_excluded=distance+4.5<=warning_radius))
command=np.asarray(f['actual_commands'][str(uid)]['waypoint']);result=dict(seed=61206,uid=uid,heat_id=2,physical_step=f['step'],origin=origin.tolist(),goal=goal.tolist(),public_warning_geometry=geometry,observed_only_goal_routes=checks,actual_command=command.tolist(),actual_command_observed_clear=rec.segment_safe(origin,command),actual_command_public_warning_clear=not haz.segment_intersects_warning_buffer(origin,command,step=f['step']),upstream_base_action_captured=False,hidden_target_position_captured=False,global_detection_impossibility_claim=False,scope='Captured observed cells and graph; public hazard geometry. Removing warning only in a counterfactual read-only route query, never a production or motion command. Actual hidden target line-of-sight and deployable peripheral sensing recovery remain unverified.',acceptance_claim=False)
with (OUT/'public_geometry_diagnosis.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
