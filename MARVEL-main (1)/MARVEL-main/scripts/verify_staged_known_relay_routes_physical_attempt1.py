"""Staged scheduler-driven saved Relay recovery; current source remains protected."""
from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));DRAFT=ROOT/'artifacts/extended_relay_known_route_design_20261013_attempt1'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt18.json').read_text());hashes={**plan['source_hashes'],**plan['pipeline_hashes']}
def guard():return all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert guard()
import integrations.gppo as package
for name,file in [('integrations.gppo.relay_frontier','relay_frontier.py'),('integrations.gppo.scheduler','scheduler.py')]:
 spec=importlib.util.spec_from_file_location(name,DRAFT/file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);setattr(package,name.rsplit('.',1)[1],module)
package.GPPOTaskScheduler=sys.modules['integrations.gppo.scheduler'].GPPOTaskScheduler
from types import SimpleNamespace
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.observed_graph import observed_next_hop
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18';frames={x['episode']:x['frame'] for x in json.loads((folder/'readonly_static64_112_control_inputs.json').read_text())};rows=[]
for episode,uid in [(64,26),(112,29)]:
 frame=frames[episode];runtime=SimulationRuntime(json.loads((folder/f'episode_{episode:03d}/gppo/config.json').read_text()));runtime.reset()
 for r in runtime.robots:
  p=frame['physical_poses_diagnostic_only'][str(r.robot_id)];r.position=np.asarray(p['position']);r.heading=p['heading'];r.travel_distance=0.
 runtime.current_step=frame['step'];robot=next(r for r in runtime.robots if r.robot_id==uid)
 rec=ObservedSearchRecovery(runtime.obstacles.frame);state=frame['observed_recovery'];rec.cells={(c,r):v for c,r,v in state['cells']};rec._authoritative_areal_grid=True;rec.prefixes={int(k):np.asarray(v) for k,v in state['prefixes'].items()};rec.prefix_headings={int(k):v for k,v in state['prefix_headings'].items()};rec.anchors={int(k):np.asarray(v) for k,v in state['anchors'].items()};rec.observe({uid:frame['selected_own_observations'][str(uid)]})
 g=frame['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)),dtype=np.int8)
 for a,b in g['zero_adjacency_edges']:adj[a,b]=0
 agent=SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
 scheduler=GPPOTaskScheduler.__new__(GPPOTaskScheduler);scheduler.runtime=runtime;scheduler._search_recovery=rec;scheduler._search_viewpoints=None;scheduler.safety=SimpleNamespace(hazards=SharedHazardAdapter(runtime));assert scheduler.__class__ is package.GPPOTaskScheduler
 anchor=np.asarray(next(a['anchor'] for a in frame['public_relay_assignments'].values() if a['helper']==uid));trace=[];collisions=0;stalled_controls=0;before_control=robot.position.copy();latest=None;reached=False
 for tick in range(600):
  if np.linalg.norm(robot.position-anchor)<=.1+1e-8:reached=True;break
  if tick%10==0:
   if tick:
    rec.observe(latest);stalled_controls=stalled_controls+1 if np.linalg.norm(robot.position-before_control)<1e-7 else 0
   before_control=robot.position.copy()
   if stalled_controls>=3:break
   action=scheduler._relay_action(robot,anchor,agent,observed_next_hop(agent,anchor),(robot.position.copy(),robot.heading))
  actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[next(i for i,r in enumerate(runtime.robots) if r.robot_id==uid)]=action
  old=robot.position.copy();latest,info=runtime.step(actions);collisions+=len(info['collisions']);trace.append(dict(tick=tick,time=runtime.current_step*runtime.dt,position=robot.position.tolist(),heading=float(robot.heading),command=[np.asarray(action[0]).tolist(),float(action[1])],moved_m=float(np.linalg.norm(robot.position-old)),feedback=runtime._motion_feedback[uid],radio=float(info['connectivity_ratio']),collisions=info['collisions']))
  if collisions:break
 row=dict(episode=episode,uid=uid,reached=reached,ticks=len(trace),control_interval_steps=10,physics_dt=runtime.dt,seconds=len(trace)*runtime.dt,collisions=collisions,stalled_controls=stalled_controls,final_position=robot.position.tolist(),final_radio=trace[-1]['radio'] if trace else None,trace=trace);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
assert guard()
result=dict(rows=rows,passed=all(r['reached'] and r['collisions']==0 for r in rows),staged_scheduler_only=True,production_unchanged=True,shield_enabled=True,radius_m=.2,coarse_graph_held_at_saved_state=True,other_uavs_held_at_saved_positions=True,observations_live_each_control=True,acceptance_claim=False)
with (DRAFT/'staged_scheduler_physical.json').open('x') as f:json.dump(result,f,indent=2)
assert result['passed']
