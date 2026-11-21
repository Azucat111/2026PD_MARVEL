"""Read-only observed radio chain diagnosis. No truth map or production change."""
from pathlib import Path
import sys,json,numpy as np
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.mesh_relay import mesh_components,ObservedMeshRelayDemandBuilder
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame
case=ROOT/'artifacts/extended_integrated_candidate_9168_30_static_20261012_attempt17/gppo'
f=json.loads((case/'mission_frames.jsonl').read_text().splitlines()[-1]);pub=json.loads((case/'joint_lifecycle_trace.json').read_text())[-1]['public_delayed_positions']
pos={int(u):np.asarray(p) for u,p in pub.items()};frame=GeometryFrame.extended(width_cells=80,height_cells=80);rec=ObservedSearchRecovery(frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in f['observed_recovery']['cells']}
points=np.array([np.array([c+.5,r+.5]) for (c,r),v in sorted(rec.cells.items()) if v==0 and rec.segment_safe(np.array([c+.5,r+.5]),np.array([c+.5,r+.5]))]);groups=mesh_components(pos,20);core=groups[0];target=groups[1]
a=np.linalg.norm(points[:,None,:]-np.array([pos[u] for u in core])[None,:,:],axis=2).min(1)<=20
z=np.linalg.norm(points[:,None,:]-np.array([pos[u] for u in target])[None,:,:],axis=2).min(1)<=20
prev={int(i):None for i in np.flatnonzero(a)};queue=deque((int(i),1) for i in np.flatnonzero(a));end=None
while queue:
 i,depth=queue.popleft()
 if z[i]:end=i;break
 if depth>=3:continue
 neighbors=np.flatnonzero(np.linalg.norm(points-points[i],axis=1)<=20)
 for j in neighbors:
  j=int(j)
  if j in prev:continue
  prev[j]=i;queue.append((j,depth+1))
chain=[]
if end is not None:
 while end is not None:chain.append(end);end=prev[end]
 chain.reverse()
coordinates=[points[i].tolist() for i in chain]
links=[]
if coordinates:
 first=np.asarray(coordinates[0]);last=np.asarray(coordinates[-1]);uid1=min(core,key=lambda u:np.linalg.norm(pos[u]-first));uid2=min(target,key=lambda u:np.linalg.norm(pos[u]-last));vertices=[pos[uid1],*map(np.asarray,coordinates),pos[uid2]];links=[float(np.linalg.norm(x-y)) for x,y in zip(vertices,vertices[1:])]
result=dict(public_component_sizes=[len(g) for g in groups],public_positions_only=True,observed_safe_anchor_count=len(points),single_observed_bridge_count=int((a&z).sum()),bounded_max_helpers=3,found_chain=bool(chain),known_safe_anchor_chain=coordinates,radio_links_m=links,all_radio_links_within20=bool(chain) and all(d<=20 for d in links),motion_paths_certified=False,helper_assignment_certified=False,production_changed=False,scope='Public observed-safe anchor and ideal radio geometry only. No truth map; this does not certify helper continuous travel or GPPO assignment feasibility.',acceptance_claim=False)
out=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17/relay_observed_chain_diagnosis_9168.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
