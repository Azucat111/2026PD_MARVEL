"""Controlled saved-state physical check; no natural/15% acceptance claim."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.search_recovery import ObservedSearchRecovery
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt18.json').read_text())
hashes={**plan['source_hashes'],**plan['pipeline_hashes']}
def source_guard():return all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert source_guard()
frames={s['episode']:s['frame'] for s in json.loads((folder/'readonly_static64_112_control_inputs.json').read_text())}
paths=json.loads((folder/'readonly_first216_low_radio_geometry_paths.json').read_text())['rows']
rows=[]
for episode,uid in [(64,26),(112,29)]:
 frame=frames[episode];path=next(r for r in paths if r['episode']==episode and r['uid']==uid)['known_grid_path']
 config=json.loads((folder/f'episode_{episode:03d}/gppo/config.json').read_text());assert not config.get('dynamic_obstacles')
 runtime=SimulationRuntime(copy.deepcopy(config));runtime.reset();assert getattr(runtime,'high_level_scheduler',None) is None
 for robot in runtime.robots:
  pose=frame['physical_poses_diagnostic_only'][str(robot.robot_id)];robot.position=np.asarray(pose['position'],dtype=float);robot.heading=float(pose['heading']);robot.travel_distance=0.
 runtime.current_step=frame['step'];robot=next(r for r in runtime.robots if int(r.robot_id)==uid)
 own=frame['selected_own_observations'][str(uid)];assert np.array_equal(robot.position,np.asarray(own['position']))
 rec=ObservedSearchRecovery(runtime.obstacles.frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in frame['observed_recovery']['cells']};rec._authoritative_areal_grid=True;rec.observe({uid:own})
 vertices=[np.asarray(p,dtype=float) for p in path];assert np.array_equal(vertices[0],robot.position)
 assert all(rec.segment_safe(a,b) for a,b in zip(vertices,vertices[1:])), 'Path must already be known at actual control snapshot'
 index=1;trace=[];collisions=0;blocked=0
 for tick in range(400):
  while index<len(vertices) and np.linalg.norm(robot.position-vertices[index])<=1e-7:index+=1
  if index>=len(vertices):break
  target=vertices[index];assert rec.segment_safe(robot.position,target)
  delta=target-robot.position;heading=float(np.degrees(np.arctan2(delta[1],delta[0]))%360)
  actions=[(r.position.copy(),r.heading) for r in runtime.robots]
  index_robot=next(i for i,r in enumerate(runtime.robots) if int(r.robot_id)==uid);actions[index_robot]=(target,heading)
  old=robot.position.copy();observations,info=runtime.step(actions);rec.observe(observations)
  moved=float(np.linalg.norm(robot.position-old));blocked+=int(moved<1e-10);collisions+=len(info['collisions'])
  trace.append(dict(tick=tick,physical_time=runtime.current_step*runtime.dt,position=robot.position.tolist(),heading=float(robot.heading),command=[target.tolist(),heading],moved_m=moved,feedback=runtime._motion_feedback[uid],collisions=info['collisions'],radio=float(info['connectivity_ratio'])))
  if collisions or sum(t['moved_m']<1e-10 for t in trace[-3:])>=3:break
 reached=float(np.linalg.norm(robot.position-vertices[-1]))<=1e-7
 row=dict(episode=episode,uid=uid,control_snapshot_seconds=frame['time'],own_observation_used=True,nearby_peer_count=len(own['nearby_robot_positions']),reached=reached,ticks=len(trace),simulated_motion_seconds=len(trace)*runtime.dt,collisions=collisions,blocked_ticks=blocked,final_position=robot.position.tolist(),final_radio=None if not trace else trace[-1]['radio'],known_path=path,trace=trace)
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('known_path','trace')}),flush=True)
assert source_guard()
result=dict(rows=rows,production_source_stable=True,unique_guarded_files=len(hashes),shield_enabled=True,radius_m=.2,controlled_saved_state_fixture=True,natural_acceptance=False,acceptance_claim=False,limitations='Other UAVs held at saved states; sensor observations and original runtime collision/Shield run live. This proves these paths in the controlled snapshot, not complete original scheduler replay or policy selection.')
with (folder/'readonly_static64_112_known_path_physical.json').open('x') as f:json.dump(result,f,indent=2)
