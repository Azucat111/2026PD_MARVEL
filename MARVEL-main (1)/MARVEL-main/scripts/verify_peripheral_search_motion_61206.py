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
BASE=ROOT/'artifacts/extended_dynamic_search_cycle_61206_20261010_attempt1';OUT=BASE/'controlled_peripheral_motion';OUT.mkdir(exist_ok=False)
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
for tick in range(600):
 while leg<len(route) and np.linalg.norm(robot.position-route[leg])<=1e-8:leg+=1
 if leg==len(route):arrived=True;detected=bool(detector.detect('oracle_only',obs,[uid]));break
 rec.observe(obs);waypoint=route[leg];heading=float(np.degrees(np.arctan2(*(goal-robot.position)[::-1]))%360.)
 assert rec.segment_safe(robot.position,waypoint)
 assert not hazards.segment_intersects_warning_buffer(robot.position,waypoint,step=runtime.current_step)
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[index]=(waypoint,heading);obs,info=runtime.step(actions);collisions.extend(info['collisions']);shield_rejections+=int(obs[uid]['motion_feedback'].get('shield_rejected',False));minimum_peer_separation=min(minimum_peer_separation,min(np.linalg.norm(robot.position-other.position) for other in runtime.robots if other.robot_id!=uid));rows.append(dict(tick=tick+1,position=robot.position.tolist(),heading=float(robot.heading),leg=leg,waypoint=waypoint.tolist(),feedback=obs[uid]['motion_feedback']))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(arrived=arrived,seconds=len(rows)*runtime.dt,position=robot.position.tolist(),heading=float(robot.heading),target_detected_by_actual_sensor=detected,collisions=collisions,shield_rejections=shield_rejections,minimum_peer_separation_m=float(minimum_peer_separation),all_segments_observed_FREE=True,all_segments_public_warning_clear=True,shield_enabled=True,physical_radius_m=.2,source_stable=True,scope='Controlled captured thirty-UAV poses with other UAVs held; true runtime motion, sensors, collision and Shield. Route chosen by offline oracle feasibility study; not deployable selection and not a natural cooperative mission.',acceptance_claim=False,passed=arrived and detected and not collisions)
(OUT/'trajectory.json').write_text(json.dumps(rows));(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['passed']
