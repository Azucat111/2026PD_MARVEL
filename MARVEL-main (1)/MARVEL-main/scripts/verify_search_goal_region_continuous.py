"""Controlled captured Search border-goal approach, continuous real motion only."""
from pathlib import Path
import json,sys,types,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.extended_travel import extended_observed_next_hop
PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt7/episode_001/gppo'
OUT=ROOT/'artifacts/extended_search_goal_region_20261010_attempt1/physical';OUT.mkdir(exist_ok=False)
frames=[json.loads(x) for x in (PAIR/'mission_frames.jsonl').read_text().splitlines()]
f=next(x for x in frames if x['step']==500);cfg=json.loads((PAIR/'config.json').read_text());poses=f['physical_poses_diagnostic_only']
for group in cfg['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[poses[str(i)]['position'] for i in range(lo,hi+1)]
runtime=SimulationRuntime(cfg);runtime.reset()
for robot in runtime.robots:robot.heading=poses[str(robot.robot_id)]['heading']
meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler);sched.runtime=runtime;sched._search_recovery=rec;sched.search_reach_radius=3.5;sched.safety=types.SimpleNamespace(hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8))
cases={}
for heat,uid in [(1,2),(2,32)]:
 g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.array(g['node_coords']);adj=np.ones((len(coords),len(coords)))
 for a,b in g['zero_adjacency_edges']:adj[a,b]=0
 agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
 robot=next(r for r in runtime.robots if r.robot_id==uid);goal=np.array(f['public_heat_positions'][str(heat)])
 cases[uid]=dict(robot=robot,agent=agent,goal=goal,arrival=None)
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for parent in ['utils','integrations'] for p in (ROOT/parent).rglob('*.py')}
(OUT/'manifest.json').write_text(json.dumps(dict(controlled_snapshot_step=500,other_uavs_hold=True,graph_frozen_from_capture=True,service_radius=3.5,physical_radius=.2,shield_enabled=True,horizon=100,decision_inputs='Captured observed graph/cells plus live own sensors and detected peers; true map only physics/sensors',not_natural_resume=True,acceptance_claim=False,source_hashes=sources),indent=2))
obs=runtime._get_observations();rows=[];collisions=[];start=time.monotonic();initial=[]
rec.observe(obs)
for uid,c in cases.items():
 details={};old=extended_observed_next_hop(uid,c['robot'].position,c['goal'],c['agent'],rec)
 new=extended_observed_next_hop(uid,c['robot'].position,c['goal'],c['agent'],rec,goal_radius=3.5,route_details=details)
 initial.append(dict(uid=uid,position=c['robot'].position.tolist(),goal=c['goal'].tolist(),heat_point_observed_FREE=rec.segment_safe(c['goal'],c['goal']),old_exact_goal_route=None if old is None else old[0].tolist(),new_service_route=None if new is None else new[0].tolist(),details=details))
 assert old is None and new is not None
(OUT/'initial_certificates.json').write_text(json.dumps(initial,indent=2))
for mission in range(1,101):
 rec.observe(obs);actions=[(r.position.copy(),r.heading) for r in runtime.robots];row=dict(mission=mission,states={})
 for uid,c in cases.items():
  robot=c['robot'];agent=c['agent'];goal=c['goal']
  if c['arrival'] is not None:continue
  agent.current_index=int(np.argmin(np.linalg.norm(agent.node_coords-robot.position,axis=1)));agent.neighbor_indices=np.flatnonzero(agent.adjacent_matrix[agent.current_index]==0)
  action=sched._search_action(robot,goal,agent,None,(robot.position.copy(),robot.heading),route_goal_radius=3.5)
  assert rec.segment_safe(robot.position,action[0])
  row['states'][uid]=dict(position=robot.position.tolist(),waypoint=action[0].tolist(),heading=float(action[1]),distance=float(np.linalg.norm(robot.position-goal)))
  actions[runtime.robots.index(robot)]=action
 for tick in range(runtime._mission_step_interval()):
  obs,info=runtime.step(actions);collisions.extend(info['collisions'])
 for uid,c in cases.items():
  if c['arrival'] is None and np.linalg.norm(c['robot'].position-c['goal'])<=3.5+1e-9:c['arrival']=mission
 rows.append(row)
 with (OUT/'trajectory.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
 if mission%10==0:print('SEARCH_REGION',mission,{uid:dict(position=c['robot'].position.tolist(),distance=float(np.linalg.norm(c['robot'].position-c['goal'])),arrival=c['arrival']) for uid,c in cases.items()},len(collisions),round(time.monotonic()-start,1),flush=True)
 if all(c['arrival'] is not None for c in cases.values()) or time.monotonic()-start>600:break
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
result=dict(cases={uid:dict(arrival_seconds=c['arrival'],final_position=c['robot'].position.tolist(),distance=float(np.linalg.norm(c['robot'].position-c['goal']))) for uid,c in cases.items()},collisions=collisions,source_stable=True,passed=all(c['arrival'] is not None for c in cases.values()) and not collisions,continuous_real_runtime=True,service_radius_unchanged=3.5,relay_exact_goal_unchanged=True,scope='Controlled border-heat approach only; no fabricated detection/service completion or natural mission acceptance.',acceptance_claim=False)
(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);assert result['passed']
