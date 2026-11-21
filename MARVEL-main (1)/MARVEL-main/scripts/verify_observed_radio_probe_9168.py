"""Bounded observed-only radio frontier prototype in a saved-pose physical fixture."""
from pathlib import Path
from collections import deque
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.mesh_relay import mesh_components,ObservedMeshRelayDemandBuilder
from utils.sensor_models import IdealSensor
case=ROOT/'artifacts/extended_integrated_candidate_9168_30_static_20261012_attempt17/gppo';out=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt1'
f=json.loads((case/'mission_frames.jsonl').read_text().splitlines()[-1]);config=json.loads((case/'config.json').read_text());runtime=SimulationRuntime(copy.deepcopy(config));runtime.reset()
assert getattr(runtime,'high_level_scheduler',None) is None
for r in runtime.robots:
 saved=f['physical_poses_diagnostic_only'][str(r.robot_id)];r.position=np.asarray(saved['position']);r.heading=float(saved['heading'])
rec=ObservedSearchRecovery(runtime.obstacles.frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in f['observed_recovery']['cells']};initial=set(rec.cells)
# Selection uses delayed public positions only; environment poses are used only by runtime.
pos={int(u):np.asarray(p) for u,p in json.loads((case/'joint_lifecycle_trace.json').read_text())[-1]['public_delayed_positions'].items()}
groups=mesh_components(pos,20);core=groups[0];remote=groups[1];uid=min(remote,key=lambda u:min(np.linalg.norm(pos[u]-pos[v]) for v in core));robot=runtime.robots[uid]
visited=set();path=[];decisions=[];trace=[];blocked=0;collision_count=0;result_reason='bounded_horizon';sensor=IdealSensor(dict(fov=120.,range=10.,los_occlusion=True),frame=rec.frame)
def select():
 free={c for c,v in rec.cells.items() if v==0};start=robot.position.copy();cell=rec.frame.world_to_cell_floor(start)
 starts=[(cell[0]+dx,cell[1]+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (cell[0]+dx,cell[1]+dy) in free and rec.segment_safe(start,np.array([cell[0]+dx+.5,cell[1]+dy+.5]))];prev={c:None for c in starts};queue=deque(starts)
 while queue:
  c=queue.popleft()
  for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
   v=(c[0]+dx,c[1]+dy)
   if v in free and v not in prev:prev[v]=c;queue.append(v)
 known=np.full((80,80),127,dtype=np.uint8)
 for (c,r),v in rec.cells.items():known[r,c]=v
 components=mesh_components(pos,20);baseline=len(components[0]);other={u:p for u,p in pos.items() if u!=uid};othergroups=mesh_components(other,20);backbone=othergroups[0];targets=[other[u] for u in backbone];choices=[]
 for c in prev:
  if c in visited or not any(rec.frame.contains_cell(c[0]+dx,c[1]+dy) and (c[0]+dx,c[1]+dy) not in rec.cells for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]):continue
  point=np.asarray([c[0]+.5,c[1]+.5]);future=mesh_components({**pos,uid:point},20)
  if len(future[0])<baseline or not all(any(set(g)-{uid}<=set(q) for q in future) for g in components):continue
  target=min(targets,key=lambda p:np.linalg.norm(point-p));bearing=float(np.degrees(np.arctan2(*(target-point)[::-1]))%360)
  potential=sum(known[r,col]==127 for col,r in sensor._visible_cells(point,bearing,known))
  if not potential:continue
  chain=[];node=c
  while node is not None:chain.append(node);node=prev[node]
  chain.reverse();vertices=[start,*[np.asarray([x+.5,y+.5]) for x,y in chain]]
  if not all(rec.segment_safe(a,b) for a,b in zip(vertices,vertices[1:])):continue
  length=sum(float(np.linalg.norm(a-b)) for a,b in zip(vertices,vertices[1:]));gap=float(np.linalg.norm(point-target));choices.append((gap+.05*length-.01*potential,c,bearing,vertices[1:],gap,int(potential)))
 if not choices:return None
 best=min(choices,key=lambda q:(q[0],q[1]));visited.add(best[1]);decisions.append(dict(uid=uid,target=list(best[1]),heading=best[2],public_gap=best[4],potential=best[5],known_cells=len(rec.cells),path=[p.tolist() for p in best[3]]));return best[3],best[2]
heading=robot.heading
for tick in range(1500):
 if not path:
  chosen=select()
  if chosen is None:result_reason='no_bounded_observed_frontier';break
  path,heading=chosen;print('PROBE',json.dumps(decisions[-1]),flush=True)
 while path and np.linalg.norm(path[0]-robot.position)<=1e-8:path.pop(0)
 if not path:continue
 target=path[0]
 if not rec.segment_safe(robot.position,target):path=[];continue
 angle=heading if len(path)==1 else float(np.degrees(np.arctan2(*(target-robot.position)[::-1]))%360)
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[uid]=(target,angle)
 old=robot.position.copy();obs,info=runtime.step(actions);rec.observe(obs);pos[uid]=np.asarray(obs[uid]['position']).copy();collision_count+=len(info['collisions']);moved=float(np.linalg.norm(old-robot.position));blocked=blocked+1 if moved<1e-10 else 0
 trace.append(dict(tick=tick,position=robot.position.tolist(),heading=robot.heading,moved_m=moved,radio=info['connectivity_ratio'],collisions=info['collisions'],known_cells=len(rec.cells)))
 if collision_count:result_reason='physical_collision';break
 if blocked>=3:path=[];blocked=0
 if tick%10==0:
  free_points=np.array([[c+.5,r+.5] for (c,r),v in rec.cells.items() if v==0 and rec.segment_safe(np.array([c+.5,r+.5]),np.array([c+.5,r+.5]))]);components=mesh_components(pos,20)
  if len(components)<2:result_reason='physically_connected';break
  a=np.array([pos[u] for u in components[0]]);b=np.array([pos[u] for g in components[1:] for u in g]);valid=(np.linalg.norm(free_points[:,None,:]-a[None,:,:],axis=2).min(axis=1)<=20)&(np.linalg.norm(free_points[:,None,:]-b[None,:,:],axis=2).min(axis=1)<=20)
  if valid.any():result_reason='observed_single_radio_bridge_available';break
result=dict(uid=uid,saved_pose_fixture=True,natural_acceptance=False,bounded_physics_ticks=1500,reason=result_reason,actual_ticks=len(trace),collisions=collision_count,new_observed_cells=len(set(rec.cells)-initial),observed_only_controller=True,shield_enabled=True,radius_m=.2,final_position=robot.position.tolist(),decisions=decisions,trace=trace,cells=[[c,r,v] for (c,r),v in sorted(rec.cells.items())],public_positions={u:p.tolist() for u,p in pos.items()},acceptance_claim=False)
(out/'observed_probe_physical_9168.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('decisions','trace','cells','public_positions')}),flush=True)
