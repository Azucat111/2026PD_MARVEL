"""Controlled continuous route fixture, never a production planner."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from utils.target_detector import TargetDetector
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
BASE=ROOT/'artifacts/extended_dynamic_search_cycle_61206_20261010_attempt1';OUT=BASE/'staged_scheduler_physical_motion';OUT.mkdir(exist_ok=False)
f=next(x for x in json.loads((BASE/'captured_frames.json').read_text()) if x['step']==400);oracle=json.loads((BASE/'peripheral_sensor_oracle.json').read_text());choice=oracle['sensor_visible_candidates'][0];cfg=json.loads((ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt8/episode_010/gppo/config.json').read_text());poses=f['physical_poses_diagnostic_only']
for group in cfg['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[poses[str(uid)]['position'] for uid in range(lo,hi+1)]
for warning in cfg['dynamic_obstacles']:
 warning['spawn_step']=0;warning['params']['initial_radius']=warning['params']['max_radius']
runtime=SimulationRuntime(cfg);runtime.reset();uid=28
for robot in runtime.robots:robot.heading=poses[str(robot.robot_id)]['heading']
robot=next(r for r in runtime.robots if r.robot_id==uid);index=runtime.robots.index(robot);meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']};hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
route=[np.asarray(p,dtype=float) for p in choice['route_waypoints']];goal=np.asarray(f['public_heat_positions']['2']);target=oracle['derived_hidden_target_oracle_only'];detector=TargetDetector({'oracle_only':[{'x':target[0],'y':target[1]}]});obs=runtime._get_observations();rows=[];collisions=[];shield_rejections=0;minimum_peer_separation=1e9;leg=0;arrived=False;detected=False
sources=list((ROOT/'utils').rglob('*.py'))+list((ROOT/'integrations').rglob('*.py'));hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
import importlib.util,types
stage=ROOT/'artifacts/extended_public_peripheral_search_design_20261011_attempt1'
for name,filename in [('integrations.gppo.search_viewpoints','search_viewpoints.py'),('integrations.gppo.scheduler','scheduler.py')]:
 spec=importlib.util.spec_from_file_location(name,stage/filename);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.search_viewpoints import ObservedSearchViewpoints
sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler);sched.runtime=runtime;sched._search_recovery=rec;sched._search_viewpoints=ObservedSearchViewpoints(rec.frame,sensor_range=10,fov=120,prior_radius=5.5);sched._search_viewpoints.checked_cells_by_heat[2]=set(map(tuple,json.loads((BASE/'own_checked_free_cells_to_step400.json').read_text())));sched._peripheral_search_targets={};sched.search_reach_radius=3.5;sched.search_assignments={2:uid};sched.safety=types.SimpleNamespace(hazards=hazards)
g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']));decisions=[];detected=False
for mission in range(80):
 sched.observe(obs);agent.current_index=int(np.argmin(np.linalg.norm(coords-robot.position,axis=1)));agent.neighbor_indices=np.flatnonzero(adj[agent.current_index]==0)
 if detector.detect('oracle_only',obs,[uid]):detected=True;break
 action=sched._search_action(robot,goal,agent,None,(robot.position.copy(),robot.heading),route_goal_radius=3.5)
 assert rec.segment_safe(robot.position,action[0]);assert not hazards.segment_intersects_warning_buffer(robot.position,action[0],step=runtime.current_step)
 decisions.append(dict(mission=mission,position=robot.position.tolist(),waypoint=action[0].tolist(),heading=float(action[1]),selected_supports={str(k):[v[0],v[1].tolist()] for k,v in sched._peripheral_search_targets.items()}))
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[index]=action
 for _ in range(runtime._mission_step_interval()):
  obs,info=runtime.step(actions);collisions.extend(info['collisions']);shield_rejections+=int(obs[uid]['motion_feedback'].get('shield_rejected',False));minimum_peer_separation=min(minimum_peer_separation,min(np.linalg.norm(robot.position-other.position) for other in runtime.robots if other.robot_id!=uid));rows.append(dict(tick=runtime.current_step,position=robot.position.tolist(),heading=float(robot.heading)))
  if detector.detect('oracle_only',obs,[uid]):detected=True;break
 if detected:break
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(seconds=len(rows)*runtime.dt,position=robot.position.tolist(),target_detected_by_actual_sensor=detected,collisions=collisions,shield_rejections=shield_rejections,minimum_peer_separation_m=float(minimum_peer_separation),public_heat_assignment_preserved=sched.search_assignments=={2:uid},all_segments_observed_FREE=True,all_segments_public_warning_clear=True,selector_hidden_target_input=False,selector_truth_map_input=False,physical_radius_m=.2,shield_enabled=True,production_source_unchanged=True,staged_source_sha256={n:hashlib.sha256((stage/n).read_bytes()).hexdigest() for n in ['scheduler.py','search_viewpoints.py']},scope='Controlled thirty-UAV captured poses; other UAVs hold. Staged production Search action -> real runtime motion -> real sensor detector, captured observed graph updated current attachment; no live MARVEL base action available, hold placeholder. Not full natural cooperative or lifecycle acceptance.',acceptance_claim=False,passed=detected and not collisions)
(OUT/'commands.json').write_text(json.dumps(decisions));(OUT/'trajectory.json').write_text(json.dumps(rows));(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['passed']
