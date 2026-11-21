"""Diagnostic observed-only peripheral Search ranking, no hidden target input."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from utils.sensor_models import IdealSensor
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.extended_travel import extended_observed_next_hop
BASE=ROOT/'artifacts/extended_dynamic_search_cycle_61206_20261010_attempt1';f=next(x for x in json.loads((BASE/'captured_frames.json').read_text()) if x['step']==400);cfg=json.loads((ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt8/episode_010/gppo/config.json').read_text());uid=28;goal=np.asarray(f['public_heat_positions']['2']);meta=f['observed_recovery']['frame'];frame=GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']);rec=ObservedSearchRecovery(frame);rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']};known=np.ones(frame.shape,dtype=np.uint8)
for (x,y),v in rec.cells.items():
 if v==0:known[y,x]=0
checked=set(map(tuple,json.loads((BASE/'own_checked_free_cells_to_step400.json').read_text())));public_prior=set((x,y) for (x,y),v in rec.cells.items() if v==0 and np.linalg.norm(frame.cell_to_world((x,y))-goal)<=cfg['_gppo_profile']['search_guaranteed_hidden_radius'])
g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']));origin=np.asarray(f['selected_own_observations'][str(uid)]['position'])
# Only the declared public warning metadata is read from this manager; never load a truth map.
om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);runtime=types.SimpleNamespace(current_step=f['step'],obstacles=om,geometry_mode='extended');haz=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8);filtered=lambda a,b:not haz.segment_intersects_warning_buffer(a,b,step=f['step']);sensor=IdealSensor(cfg['sensor']['params'],frame=frame);rows=[]
for point in coords:
 if np.linalg.norm(point-goal)>20 or not rec.segment_safe(point,point) or not filtered(point,point):continue
 heading=float(np.degrees(np.arctan2(*(goal-point)[::-1]))%360.);visible=set(map(tuple,sensor._visible_cells(point,heading,known)));gain=len((visible&public_prior)-checked)
 if gain==0:continue
 details={};hop=extended_observed_next_hop(uid,origin,point,agent,rec,segment_filter=filtered,route_details=details)
 if hop is None:continue
 route=[origin]+[np.array(p) for p in details['route_waypoints']];cost=sum(float(np.linalg.norm(b-a)) for a,b in zip(route,route[1:]));score=gain/max(1,cost);rows.append(dict(pose=point.tolist(),public_heading=heading,unchecked_observed_FREE_prior_cells=gain,observed_route_length_m=cost,diagnostic_gain_per_route_m=score,route_waypoints=details['route_waypoints']))
rows.sort(key=lambda r:(-r['diagnostic_gain_per_route_m'],r['pose']));result=dict(seed=61206,uid=uid,heat_id=2,captured_step=400,public_prior_radius_m=cfg['_gppo_profile']['search_guaranteed_hidden_radius'],known_prior_FREE_cells=len(public_prior),own_checked_prior_cells=len(public_prior&checked),hidden_target_input=False,truth_map_input=False,unknown_LOS_cells_opaque=True,observed_only_ranked_candidates=rows,scope='Read-only bounded design feasibility using current sensor primitive over observed labels only and existing Dijkstra. Ranking is not integrated in production; continuous dynamic peer and full natural mission validation pending.',acceptance_claim=False)
with (BASE/'observed_only_peripheral_ranking.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(candidates=len(rows),top=rows[:3],hidden_target_input=False,truth_map_input=False)))
