"""Observed-only bounded physical escape diagnostic; no production planner."""
from pathlib import Path
import sys,json
from collections import deque
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from utils.dynamics_models import KinematicSimple
from integrations.gppo.search_recovery import ObservedSearchRecovery
OUT=ROOT/'artifacts/extended_warning_residual_snapshot_20261015_attempt23'
raw=json.loads((OUT/'snapshots.json').read_text());snapshots={r['step']:r for r in raw};reports=[]
for step,row in sorted(snapshots.items()):
 f=row['frame'];frame=GeometryFrame(f['origin'][0],f['origin'][1],f['cell_size'],f['width_cells'],f['height_cells'],mode=f['mode'],extent_x=f['extent_m'][0],extent_y=f['extent_m'][1])
 rec=ObservedSearchRecovery(frame);rec.cells={(c,r):v for c,r,v in row['cells']}
 start=np.array(row['position']);model=KinematicSimple(dict(velocity=row['model_velocity']));dt=row['dt'];spacing=model.velocity*dt
 def radius(h,t):
  return min(h['max_radius'],h['initial_radius']+max(0,t-h['spawn_step'])*h['expansion_rate'])
 def clear(a,b,elapsed):
  if not rec.segment_safe(a,b):return False
  direction=b-a;denom=float(direction@direction)
  for peer in row['peers']:
   p=np.array(peer);fraction=float(np.clip((p-a)@direction/denom,0,1)) if denom>1e-20 else 0
   if np.linalg.norm(a+fraction*direction-p)<min(.5,float(np.linalg.norm(start-p)))-1e-9:return False
  for h in row['hazards']:
   if step+elapsed<h['spawn_step']:continue
   c=np.array(h['center']);fraction=float(np.clip((c-a)@direction/denom,0,1)) if denom>1e-20 else 0
   if np.linalg.norm(a+fraction*direction-c)<=radius(h,step+elapsed)+.2+1e-9:return False
  return True
 def goal(p):return all(np.linalg.norm(p-np.array(h['center']))>h['max_radius']+.4 for h in row['hazards'])
 first=[];directions=[(1,0),(-1,0),(0,1),(0,-1)]
 for dx,dy in directions:
  target=start+spacing*np.array([dx,dy]);state,_=model.step(start,target,row['heading'],0,row['velocity'],dt)
  if clear(start,state['position'],1):first.append(dict(target=target.tolist(),proposal=state['position'].tolist()))
 queue=deque([(0,0)]);parent={(0,0):None};depth={(0,0):0};endpoint=None
 while queue:
  key=queue.popleft();n=depth[key];p=start+spacing*np.array(key)
  if goal(p):endpoint=key;break
  if n>=60:continue
  for dx,dy in directions:
   nxt=(key[0]+dx,key[1]+dy)
   if nxt in parent or max(abs(nxt[0]),abs(nxt[1]))>25:continue
   q=start+spacing*np.array(nxt)
   if clear(p,q,n+1):parent[nxt]=key;depth[nxt]=n+1;queue.append(nxt)
 path=[]
 if endpoint is not None:
  while endpoint is not None:path.append(endpoint);endpoint=parent[endpoint]
  path.reverse()
 actual=start.copy();yaw=row['heading'];validated=[]
 for index,key in enumerate(path[1:],1):
  target=start+spacing*np.array(key);d=target-actual;heading=float(np.degrees(np.arctan2(d[1],d[0]))%360)
  state,_=model.step(actual,target,yaw,heading,row['velocity'],dt);proposal=state['position']
  assert clear(actual,proposal,index)
  validated.append(dict(tick=step+index,target=target.tolist(),heading=heading,proposal=proposal.tolist()))
  actual=proposal;yaw=state['heading']
 reports.append(dict(step=step,position=start.tolist(),online_selected_target=row['selected_target'],legal_observed_first_steps=first,bounded_path_found=bool(path),actions=validated,physical_validation=bool(path),bounds=dict(radius_m=2.5,max_ticks=60,resolution_m=spacing,headings=4),known_cells_only=True,peer_limit='Peers held at snapshot positions; re-observe and recertify on every live tick',goal='outside final public no-fly radius +0.4m, not full 8m Safety release'))
result=dict(diagnostic_only=True,production_modified=False,acceptance_claim=False,reports=reports)
(OUT/'observed_escape_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps([dict(step=r['step'],legal_first=len(r['legal_observed_first_steps']),path=r['bounded_path_found'],ticks=len(r['actions'])) for r in reports]),flush=True)
