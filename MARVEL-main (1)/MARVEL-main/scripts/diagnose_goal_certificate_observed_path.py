"""Captured observed-only motion decision; true map is validation oracle only."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.extended_travel import extended_observed_next_hop,extended_observed_travel_time_matrix
from integrations.gppo.observed_graph import observed_next_hop
from integrations.gppo.safety_layer import SharedHazardAdapter
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from utils.dynamics_models import KinematicSimple
PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt6/episode_002/gppo'
OUT=ROOT/'artifacts/extended_goal_route_certificate_20261010_attempt1/boundary_after.json'
frames=[json.loads(s) for s in (PAIR/'mission_frames.jsonl').read_text().splitlines()]
f=next(x for x in frames if x['step']==560);prior=next(x for x in frames if x['step']==550);uid=15
meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
rec.observe({int(k):o for k,o in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
for field in ('prefixes','prefix_headings','anchors'):
    setattr(rec,field,{int(k):float(v) if field=='prefix_headings' else np.array(v) for k,v in prior['observed_recovery'][field].items()})
rec.failed={int(k):[(np.array(t),np.array(p)) for t,p in values] for k,values in prior['observed_recovery']['failed'].items()}
g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
own=f['selected_own_observations'][str(uid)];robot=types.SimpleNamespace(robot_id=uid,position=np.array(own['position']),heading=own['heading'],current_task='relay')
anchor=np.array(next(a['anchor'] for a in f['public_relay_assignments'].values() if a['helper']==uid))
cfg=json.loads((PAIR/'config.json').read_text());om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);om.step(560,.1)
runtime=types.SimpleNamespace(current_step=560,obstacles=om,geometry_mode='extended',robots=[robot]);hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
filtered=lambda a,b:not hazards.segment_intersects_warning_buffer(a,b,step=560)
# Offline diagnostic only: use observed FREE cell centres, never production decisions.
import heapq
points={c:np.array(rec.frame.origin)+(np.array(c)+.5)*rec.frame.cell_size for c,v in rec.cells.items() if v==0}
dist={};previous={};heap=[]
for c,p in points.items():
 if np.linalg.norm(p-robot.position)<=2 and rec.segment_safe(robot.position,p):
  dist[c]=float(np.linalg.norm(p-robot.position));heapq.heappush(heap,(dist[c],c))
end=None
while heap:
 d,c=heapq.heappop(heap)
 if d!=dist[c]:continue
 p=points[c]
 if np.linalg.norm(p-anchor)<=2 and rec.segment_safe(p,anchor):end=c;break
 for delta in ((1,0),(-1,0),(0,1),(0,-1)):
  other=(c[0]+delta[0],c[1]+delta[1])
  if other not in points or not rec.segment_safe(p,points[other]):continue
  cost=d+rec.frame.cell_size
  if cost<dist.get(other,float('inf')):
   dist[other]=cost;previous[other]=c;heapq.heappush(heap,(cost,other))
route=[];node=end
while node is not None:
 route.append(points[node].tolist());node=previous.get(node)
result=dict(observed_cells=len(rec.cells),observed_FREE_centres=len(points),reachable_centres=len(dist),found=end is not None,known_path_m=None if end is None else dist[end],route=list(reversed(route)),scope='Offline observed-map cell-centre search, not a production planner; failure limited to this discretisation.',acceptance_claim=False)
with (ROOT/'artifacts/extended_goal_route_certificate_20261010_attempt1/observed_cell_path_diagnosis.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='route'}))
