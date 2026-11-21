"""Observed-only legal radio frontier candidate diagnosis; not a production planner."""
from pathlib import Path
from collections import deque
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from utils.sensor_models import IdealSensor
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.mesh_relay import mesh_components
case=ROOT/'artifacts/extended_integrated_candidate_9168_30_static_20261012_attempt17/gppo'
f=json.loads((case/'mission_frames.jsonl').read_text().splitlines()[-1]);pub=json.loads((case/'joint_lifecycle_trace.json').read_text())[-1]['public_delayed_positions'];pos={int(u):np.asarray(p) for u,p in pub.items()}
frame=GeometryFrame.extended(width_cells=80,height_cells=80);rec=ObservedSearchRecovery(frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in f['observed_recovery']['cells']}
free={c for c,v in rec.cells.items() if v==0};frontiers={c for c in free if any(frame.contains_cell(c[0]+dx,c[1]+dy) and (c[0]+dx,c[1]+dy) not in rec.cells for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])}
known=np.full((80,80),127,dtype=np.uint8)
for (c,r),v in rec.cells.items():known[r,c]=v
sensor=IdealSensor(dict(fov=120.,range=10.,los_occlusion=True),frame=frame);groups=mesh_components(pos,20);core=groups[0];remote=groups[1];baseline=len(core);candidates=[]
for uid in core:
 start=pos[uid];c0=frame.world_to_cell_floor(start);starts=[(c0[0]+dx,c0[1]+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (c0[0]+dx,c0[1]+dy) in free and rec.segment_safe(start,np.array([c0[0]+dx+.5,c0[1]+dy+.5]))]
 prev={c:None for c in starts};queue=deque(starts)
 while queue:
  c=queue.popleft()
  for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
   v=(c[0]+dx,c[1]+dy)
   if v in free and v not in prev:prev[v]=c;queue.append(v)
 for cell in frontiers & prev.keys():
  point=np.array([cell[0]+.5,cell[1]+.5]);forecast=mesh_components({**pos,uid:point},20)
  if len(forecast[0])<baseline or not all(any(set(g)-{uid}<=set(q) for q in forecast) for g in groups):continue
  target=min(remote,key=lambda u:np.linalg.norm(point-pos[u]));direction=pos[target]-point;heading=float(np.degrees(np.arctan2(direction[1],direction[0]))%360)
  visible=sensor._visible_cells(point,heading,known);potential=sum(int(known[r,c])==127 for c,r in visible)
  if not potential:continue
  chain=[];node=cell
  while node is not None:chain.append(node);node=prev[node]
  chain.reverse();vertices=[start,*[np.array([c+.5,r+.5]) for c,r in chain]]
  if not all(rec.segment_safe(a,b) for a,b in zip(vertices,vertices[1:])):continue
  length=sum(float(np.linalg.norm(a-b)) for a,b in zip(vertices,vertices[1:]));gap=float(np.linalg.norm(point-pos[target]));oldgap=min(float(np.linalg.norm(start-pos[t])) for t in remote)
  if gap>=oldgap-1e-9:continue
  candidates.append(dict(uid=uid,anchor=point.tolist(),target_boundary=target,public_gap_m=gap,old_gap_m=oldgap,unknown_potential_cells=potential,known_certified_path_m=length,forecast_lcc=len(forecast[0]),path=[p.tolist() for p in vertices],heading=heading,score=gap+.01*length-.01*potential))
candidates.sort(key=lambda x:(x['score'],x['uid'],x['anchor']));result=dict(observed_only=True,frontier_count=len(frontiers),candidates=len(candidates),examples=candidates[:3],continuous_known_five_probe_segments_verified=bool(candidates),current_production_allows_lcc_equal=False,potential_sensing_is_uncertain=True,production_changed=False,scope='Observed known-free continuous segment certificates and public radio forecast only. Actual hidden cells and peers still require real sensing/Shield validation. No guarantee that a radio bridge can be completed.',acceptance_claim=False)
(ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17/relay_observed_frontier_diagnosis_9168.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'examples':[{k:v for k,v in c.items() if k!='path'} for c in candidates[:3]]}))
