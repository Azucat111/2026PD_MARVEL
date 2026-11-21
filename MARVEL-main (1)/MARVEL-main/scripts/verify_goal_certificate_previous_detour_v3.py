"""Controlled continuous recovery of captured Relay cycle; no natural-state resume claim."""
from pathlib import Path
import sys,json,time,hashlib,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.observed_graph import observed_next_hop
PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt4/episode_002/gppo';OUT=ROOT/'artifacts/extended_goal_certificate_previous_detour_20261010_attempt3';OUT.mkdir(exist_ok=False)
frames=[json.loads(s) for s in (PAIR/'mission_frames.jsonl').read_text().splitlines()];f=next(x for x in frames if x['step']==480);prior=next(x for x in frames if x['step']==470);cfg=json.loads((PAIR/'config.json').read_text())
poses=f['physical_poses_diagnostic_only']
for group in cfg['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[poses[str(uid)]['position'] for uid in range(lo,hi+1)]
# Controlled fixture starts with the same public warning field already at its captured cap.
for warning in cfg['dynamic_obstacles']:
 warning['spawn_step']=0;warning['params']['initial_radius']=warning['params']['max_radius']
runtime=SimulationRuntime(cfg);runtime.reset()
for robot in runtime.robots:robot.heading=poses[str(robot.robot_id)]['heading']
robot=next(r for r in runtime.robots if r.robot_id==21);anchor=np.array(f['public_relay_assignments']['6']['anchor'])
meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
for field in ('prefixes','prefix_headings','anchors'):
 setattr(rec,field,{int(uid):float(v) if field=='prefix_headings' else np.array(v) for uid,v in prior['observed_recovery'][field].items()})
rec.failed={int(uid):[(np.array(t),np.array(p)) for t,p in values] for uid,values in prior['observed_recovery']['failed'].items()}
g=f['observed_graphs_diagnostic_only']['21'];coords=np.array(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler);sched.runtime=runtime;sched._search_recovery=rec;sched.search_reach_radius=3.5;sched.safety=types.SimpleNamespace(hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8))
sources=list((ROOT/'utils').rglob('*.py'))+list((ROOT/'integrations').rglob('*.py'))+[Path(__file__)]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(OUT/'manifest.json').write_text(json.dumps(dict(controlled_initial_poses_from_capture=True,not_natural_resume=True,observed_graph_frozen_from_capture=True,public_warning_initial_cap=4,other_uavs_hold=True,helper=21,original_anchor=anchor.tolist(),horizon=180,wall_budget=600,source_hashes=hashes,decision_inputs='historically observed cells, current own sensors, observed node graph, public warning; true map only inside real physics/sensors',shield_enabled=True,physical_radius=.2,acceptance_claim=False),indent=2))
(OUT/'config.json').write_text(json.dumps(cfg,indent=2));obs=runtime._get_observations();rows=[];collisions=[];arrival=None;start=time.monotonic()
for mission in range(1,181):
 rec.observe(obs);agent.current_index=int(np.argmin(np.linalg.norm(coords-robot.position,axis=1)));agent.neighbor_indices=np.flatnonzero(adj[agent.current_index]==0)
 action=sched._relay_action(robot,anchor,agent,observed_next_hop(agent,anchor),(robot.position.copy(),robot.heading))
 row=dict(mission=mission,position=robot.position.tolist(),waypoint=action[0].tolist(),heading=action[1],observed_segment_FREE=rec.segment_safe(robot.position,action[0]),warning_clear=not sched.safety.hazards.segment_intersects_warning_buffer(robot.position,action[0],step=runtime.current_step),distance=float(np.linalg.norm(robot.position-anchor)))
 assert row['observed_segment_FREE'] and row['warning_clear']
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[runtime.robots.index(robot)]=action
 for _ in range(runtime._mission_step_interval()):
  obs,info=runtime.step(actions);collisions.extend(info['collisions'])
 row['end_position']=robot.position.tolist();row['shield_rejected']=obs[21].get('motion_feedback',{}).get('shield_rejected');rows.append(row)
 with (OUT/'trajectory.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
 if mission%10==0:print('RECOVERY',mission,row['end_position'],row['distance'],len(collisions),round(time.monotonic()-start,1),flush=True)
 if np.linalg.norm(robot.position-anchor)<=.1:arrival=mission;break
 if time.monotonic()-start>600:break
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
summary=dict(arrival_mission=arrival,final_position=robot.position.tolist(),final_distance=float(np.linalg.norm(robot.position-anchor)),collisions=collisions,steps=len(rows),all_commanded_segments_observed_FREE=True,all_commands_public_warning_clear=True,source_hashes_preserved=True,continuous_runtime_no_teleport=True,shield_enabled=True,scope='Controlled snapshot geometry with stationary other UAVs, real physics/sensors and original fixed Relay goal; not a full natural cooperative mission.',passed=arrival is not None and not collisions,acceptance_claim=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
assert summary['passed']
