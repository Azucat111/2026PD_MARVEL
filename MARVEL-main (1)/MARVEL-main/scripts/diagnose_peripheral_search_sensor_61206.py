"""Offline sensor oracle; hidden truth never enters a production decision."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from utils.sensor_models import IdealSensor
from integrations.gppo.search_scenario import build_search_scenario,FrozenSearchRadii
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.extended_travel import extended_observed_next_hop
PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt8/episode_010/gppo';OUT=ROOT/'artifacts/extended_dynamic_search_cycle_61206_20261010_attempt1'
f=next(x for x in json.loads((OUT/'captured_frames.json').read_text()) if x['step']==400);cfg=json.loads((PAIR/'config.json').read_text());initial=json.loads((PAIR/'initial.json').read_text());uid=28
om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);grid=om.get_occupancy_grid()
heat,survivors,_=build_search_scenario(occupancy_grid=grid,robot_locations=[np.array(v) for v in initial['initial_positions']],seed=cfg['task_scheduler']['search_scenario_seed'],radii=FrozenSearchRadii.derive(sensor_range=10,cell_size=1,search_service_steps=6),origin=om.frame.origin)
assert [dict(id=h.heat_id,position=list(h.position)) for h in heat]==initial['public_heats']
target=next(s for s in survivors if s.heat_id==2);target_cell=tuple(map(int,np.rint(target.position)));goal=np.asarray(f['public_heat_positions']['2']);meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
origin=np.asarray(f['selected_own_observations'][str(uid)]['position']);om.step(f['step'],.1);runtime=types.SimpleNamespace(current_step=f['step'],obstacles=om,geometry_mode='extended');haz=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8);filtered=lambda a,b:not haz.segment_intersects_warning_buffer(a,b,step=f['step']);sensor=IdealSensor(cfg['sensor']['params'],frame=om.frame)
count=0;candidates=[]
for point in coords:
 if np.linalg.norm(point-goal)>20 or not rec.segment_safe(point,point) or not filtered(point,point) or om.check_collision(point,radius=.2)[0]:continue
 count+=1;heading=float(np.degrees(np.arctan2(*(goal-point)[::-1]))%360.);visible=set(map(tuple,sensor._visible_cells(point,heading,grid)))
 if target_cell not in visible:continue
 details={};hop=extended_observed_next_hop(uid,origin,point,agent,rec,segment_filter=filtered,route_details=details)
 candidates.append(dict(pose=point.tolist(),public_heat_heading=heading,actual_target_visible_sensor_oracle=True,observed_route_exists=hop is not None,next_hop=None if hop is None else hop[0].tolist(),route_waypoints=details.get('route_waypoints'),route_goal_connected=details.get('route_goal_connected')))
result=dict(seed=61206,uid=uid,heat_id=2,derived_hidden_target_oracle_only=list(target.position),truth_provenance='Deterministic source-stable Search generator from stored static map and exact initial positions; all four public heats exactly matched. Hidden position was not directly captured.',candidate_domain='Existing captured observed graph nodes within20m of public heat; point five-probe0.2m physical, observed FREE and public warning checks. Sensor heading aims at public heat, not hidden target.',legal_points_examined=count,sensor_visible_candidates=candidates,physical_continuous_path_and_peer_execution_verified=False,deployability='Candidate positions and public headings are observable; hidden visibility was used solely to classify oracle feasibility, never to select a production action. A deployable selection/coverage memory rule is still missing.',global_impossibility_claim=False,acceptance_claim=False)
with (OUT/'peripheral_sensor_oracle.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(legal_points_examined=count,visible_candidates=len(candidates),observed_routes=sum(c['observed_route_exists'] for c in candidates),sample=candidates[:3])))
