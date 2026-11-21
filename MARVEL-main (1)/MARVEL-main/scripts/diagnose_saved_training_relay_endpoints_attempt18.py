"""Geometry-only observed-map diagnosis of saved Relay endpoints; never production."""
from pathlib import Path
import sys,json,hashlib
from collections import deque
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_travel import extended_observed_next_hop
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18'
source=folder/'readonly_first216_low_radio_snapshot_inputs.json'
rows=[]
for snap in json.loads(source.read_text()):
 desc=snap['frame'];frame=GeometryFrame.extended(width_cells=desc['width_cells'],height_cells=desc['height_cells'],cell_size=desc['cell_size'])
 assert frame.describe()==desc
 rec=ObservedSearchRecovery(frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in snap['cells']};rec._authoritative_areal_grid=True
 free={c for c,v in rec.cells.items() if v==0}
 def centre(c):return np.asarray(frame.origin)+(np.asarray(c,dtype=float)+.5)*frame.cell_size
 for relay in snap['relays']:
  uid=int(relay['helper']);start=np.asarray(relay['own_position_diagnostic'],dtype=float);goal=np.asarray(relay['anchor'],dtype=float);rec.observed_positions={uid:start}
  agent=None;details={};graph=relay['graph']
  if graph is not None:
   coords=np.asarray(graph['node_coords']);adj=np.ones((len(coords),len(coords)),dtype=np.int8)
   for a,b in graph['zero_adjacency_edges']:adj[int(a),int(b)]=0
   agent=SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=graph['current_index'],neighbor_indices=np.asarray(graph['neighbor_indices']))
  hop=extended_observed_next_hop(uid,start,goal,agent,rec,route_details=details)
  c0=frame.world_to_cell_floor(start)
  starts=[(c0[0]+dx,c0[1]+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (c0[0]+dx,c0[1]+dy) in free and rec.segment_safe(start,centre((c0[0]+dx,c0[1]+dy)))]
  prev={c:None for c in starts};q=deque(starts);finish=None
  if np.linalg.norm(goal-start)<=.1:finish='arrived'
  elif rec.segment_safe(start,goal):finish='direct'
  else:
   while q:
    cell=q.popleft();point=centre(cell)
    if np.linalg.norm(point-goal)<=1.5 and rec.segment_safe(point,goal):finish=cell;break
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
     other=cell[0]+dx,cell[1]+dy
     if other in free and other not in prev and rec.segment_safe(point,centre(other)):
      prev[other]=cell;q.append(other)
  path=[]
  if finish in ('arrived','direct'):path=[start,goal]
  elif finish is not None:
   chain=[];c=finish
   while c is not None:chain.append(centre(c));c=prev[c]
   path=[start,*reversed(chain),goal]
  validated=bool(path) and all(rec.segment_safe(a,b) for a,b in zip(path,path[1:]))
  rows.append(dict(episode=snap['episode'],seed=snap['seed'],uid=uid,radio=snap['radio'],start=start.tolist(),goal=goal.tolist(),coarse_next_hop=None if hop is None else np.asarray(hop[0]).tolist(),coarse_route_details=details,known_grid_path_found=bool(path),known_grid_path_segments_validated=validated,known_grid_path_m=sum(float(np.linalg.norm(a-b)) for a,b in zip(path,path[1:])),known_grid_path=[p.tolist() for p in path],visited_cells=len(prev),bound_cells=frame.width_cells*frame.height_cells,grid_resolution_m=frame.cell_size,radius_m=.2,missing=snap['missing']))
result=dict(rows=rows,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Saved observed occupancy and coarse graph; static geometry-only diagnosis. Missing peers, current public warning geometry and controller history. Paths are not full runtime/SafetyShield deployment certificates.',acceptance_claim=False)
with (folder/'readonly_first216_low_radio_geometry_paths.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps([{k:r[k] for k in ('episode','uid','coarse_next_hop','known_grid_path_found','known_grid_path_segments_validated','known_grid_path_m','visited_cells')} for r in rows]),flush=True)
