"""Use the existing observed mesh builder after bounded sensing, certify physical deployment."""
from pathlib import Path
from collections import deque
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder,mesh_components
out=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt1';p=json.loads((out/'observed_probe_physical_9168.json').read_text());case=ROOT/'artifacts/extended_integrated_candidate_9168_30_static_20261012_attempt17/gppo';f=json.loads((case/'mission_frames.jsonl').read_text().splitlines()[-1]);runtime=SimulationRuntime(json.loads((case/'config.json').read_text()));runtime.reset()
for r in runtime.robots:
 saved=f['physical_poses_diagnostic_only'][str(r.robot_id)];r.position=np.asarray(saved['position']);r.heading=float(saved['heading'])
runtime.robots[p['uid']].position=np.asarray(p['final_position']);runtime.robots[p['uid']].heading=p['trace'][-1]['heading']
rec=ObservedSearchRecovery(runtime.obstacles.frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in p['cells']};pos={int(u):np.asarray(x) for u,x in p['public_positions'].items()};free={c for c,v in rec.cells.items() if v==0}
builder=ObservedMeshRelayDemandBuilder();builder.anchor_candidates_provider=lambda:np.array([[c+.5,r+.5] for c,r in free if rec.segment_safe(np.array([c+.5,r+.5]),np.array([c+.5,r+.5]))]);demand=builder.build(pos);choices=[]
for slot in demand.slots:
 anchor=np.asarray(slot.position);cell=rec.frame.world_to_cell_floor(anchor)
 for uid,start in pos.items():
  if uid in slot.forbidden_uav_ids:continue
  original=mesh_components(pos,20);forecast=mesh_components({**pos,uid:anchor},20)
  if len(forecast[0])<=len(original[0]) or not all(any(set(g)-{uid}<=set(q) for q in forecast) for g in original):continue
  c0=rec.frame.world_to_cell_floor(start);starts=[(c0[0]+dx,c0[1]+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (c0[0]+dx,c0[1]+dy) in free and rec.segment_safe(start,np.array([c0[0]+dx+.5,c0[1]+dy+.5]))];prev={c:None for c in starts};q=deque(starts)
  while q and cell not in prev:
   c=q.popleft()
   for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    v=(c[0]+dx,c[1]+dy)
    if v in free and v not in prev:prev[v]=c;q.append(v)
  if cell not in prev:continue
  chain=[];node=cell
  while node is not None:chain.append(node);node=prev[node]
  chain.reverse();vertices=[start,*[np.asarray([c+.5,r+.5]) for c,r in chain]]
  if not all(rec.segment_safe(a,b) for a,b in zip(vertices,vertices[1:])):continue
  length=sum(float(np.linalg.norm(a-b)) for a,b in zip(vertices,vertices[1:]));choices.append((length,uid,slot.source_entity_id,vertices[1:],len(forecast[0])))
choices.sort(key=lambda c:c[:3]);rows=[]
# Each candidate is an independent saved-pose fixture; avoid counting diagnostic failures as production acceptance.
for choice in choices[:3]:
 for r in runtime.robots:r.position=np.asarray(p['public_positions'][str(r.robot_id)]).copy();r.heading=float(f['physical_poses_diagnostic_only'][str(r.robot_id)]['heading'])
 length,uid,target,path,forecast=choice;robot=runtime.robots[uid];index=0;trace=[];blocked=0;collisions=0;stable=0
 for tick in range(500):
  while index<len(path) and np.linalg.norm(path[index]-robot.position)<=1e-8:index+=1
  waypoint=path[index] if index<len(path) else robot.position.copy();angle=robot.heading if index>=len(path) else float(np.degrees(np.arctan2(*(waypoint-robot.position)[::-1]))%360)
  assert rec.segment_safe(robot.position,waypoint)
  actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[uid]=(waypoint,angle);old=robot.position.copy();obs,info=runtime.step(actions);moved=float(np.linalg.norm(old-robot.position));collisions+=len(info['collisions']);blocked=blocked+1 if moved<1e-10 and index<len(path) else 0;stable=stable+1 if info['connectivity_ratio']>=.9 else 0
  trace.append(dict(tick=tick,position=robot.position.tolist(),heading=robot.heading,radio=info['connectivity_ratio'],moved_m=moved,collisions=info['collisions'],feedback=runtime._motion_feedback[uid]))
  if stable>=30 or collisions or blocked>=3:break
 rows.append(dict(uid=uid,target=target,known_path_m=length,path=[x.tolist() for x in path],forecast_lcc=forecast,reached=index>=len(path),stable_radio_seconds=stable*runtime.dt,collisions=collisions,blocked_ticks=blocked,trace=trace));print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('trace','path')}),flush=True)
 if stable>=30 and not collisions:break
result=dict(existing_builder_slots=len(demand.slots),observed_motion_candidates=len(choices),rows=rows,saved_pose_fixture=True,physical_recovery=any(r['stable_radio_seconds']>=3 and not r['collisions'] for r in rows),observed_only_selection=True,shield_enabled=True,radius=.2,production_unchanged=True,acceptance_claim=False)
(out/'existing_mesh_bridge_after_probe_9168.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
