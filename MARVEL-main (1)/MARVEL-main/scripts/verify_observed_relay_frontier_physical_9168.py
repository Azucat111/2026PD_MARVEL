"""Saved-pose observed frontier physical validation, not natural mission acceptance."""
from pathlib import Path
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame
out=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17'
case=ROOT/'artifacts/extended_integrated_candidate_9168_30_static_20261012_attempt17/gppo'
diagnosis=json.loads((out/'relay_observed_frontier_diagnosis_9168.json').read_text())
frame=json.loads((case/'mission_frames.jsonl').read_text().splitlines()[-1])
config=json.loads((case/'config.json').read_text());assert not config.get('dynamic_obstacles')
rows=[]
for candidate in diagnosis['examples']:
 runtime=SimulationRuntime(copy.deepcopy(config));runtime.reset()
 assert getattr(runtime,'high_level_scheduler',None) is None
 for r in runtime.robots:
  saved=frame['physical_poses_diagnostic_only'][str(r.robot_id)];r.position=np.asarray(saved['position']);r.heading=float(saved['heading']);r.travel_distance=0.
 uid=candidate['uid'];robot=next(r for r in runtime.robots if r.robot_id==uid)
 rec=ObservedSearchRecovery(runtime.obstacles.frame);rec.cells={(int(c),int(r)):int(v) for c,r,v in frame['observed_recovery']['cells']};before=set(rec.cells)
 vertices=[np.asarray(p) for p in candidate['path']];index=1;trace=[];collision_count=0;blocked=0;min_radio=1.
 for tick in range(250):
  if index>=len(vertices):break
  target=vertices[index];assert rec.segment_safe(robot.position,target)
  direction=target-robot.position;heading=float(np.degrees(np.arctan2(direction[1],direction[0]))%360)
  # Final segment aims sensor toward public disconnected boundary; dynamics still enforces heading.
  if index==len(vertices)-1:heading=candidate['heading']
  actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[uid]=(target,heading)
  old=robot.position.copy();obs,info=runtime.step(actions);feedback=runtime._motion_feedback[uid]
  collision_count+=len(info['collisions']);min_radio=min(min_radio,info['connectivity_ratio'])
  moved=float(np.linalg.norm(robot.position-old));blocked+=int(moved<1e-10)
  observed=obs[uid]
  trace.append(dict(tick=tick,position=robot.position.tolist(),heading=robot.heading,command=[target.tolist(),heading],moved_m=moved,feedback=feedback,connectivity=info['connectivity_ratio'],collisions=info['collisions']))
  if np.linalg.norm(robot.position-target)<=1e-8:index+=1
  if blocked>=3:break
 sensor=runtime.sensor_by_type.get(robot.robot_type,runtime.sensor)
 visible=sensor._visible_cells(robot.position,robot.heading,runtime.obstacles.get_occupancy_grid())
 new_observed=[dict(cell=[int(c),int(r)],value=int(runtime.obstacles.get_occupancy_grid()[r,c])) for c,r in visible if (int(c),int(r)) not in before]
 rows.append(dict(uid=uid,reached=index>=len(vertices),ticks=len(trace),collisions=collision_count,blocked_ticks=blocked,final_position=robot.position.tolist(),final_heading=robot.heading,min_physical_connectivity=min_radio,new_actual_sensor_cells=len(new_observed),new_cells=new_observed,trace=trace))
 print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('trace','new_cells')}),flush=True)
result=dict(saved_pose_controlled_fixture=True,natural_rollout=False,controller_inputs='saved observed cells, own pose/heading and delayed public peer positions',environment_truth_used_only_for='restoring saved physical state, runtime collision and actual sensing',radius_m=.2,shield_enabled=True,production_changed=False,rows=rows,acceptance_claim=False)
(out/'relay_observed_frontier_physical_9168.json').write_text(json.dumps(result,indent=2))
