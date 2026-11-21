"""Staged production scheduler -> adapter -> real motion, saved-pose Relay fixture."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));STAGE=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt1'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt17.json').read_text());hashes={**plan['source_hashes'],**plan['pipeline_hashes']}
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
for name,file in [('integrations.gppo.relay_frontier','relay_frontier.py'),('integrations.gppo.scheduler','scheduler_draft.py'),('utils.policy_adapter','policy_adapter_draft.py')]:
 spec=importlib.util.spec_from_file_location(name,STAGE/file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);parent,attr=name.rsplit('.',1);setattr(sys.modules[parent],attr,module)
 if name=='utils.policy_adapter':module._MARVEL_ROOT=ROOT
from utils.simulation_runtime import SimulationRuntime
from utils.policy_adapter import MARVELPolicyAdapter
case=ROOT/'artifacts/extended_integrated_candidate_9168_30_static_20261012_attempt17/gppo';f=json.loads((case/'mission_frames.jsonl').read_text().splitlines()[-1]);config=json.loads((case/'config.json').read_text())
for group in config['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[f['physical_poses_diagnostic_only'][str(u)]['position'] for u in range(lo,hi+1)]
checkpoint=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17/checkpoint_batch_015.pth';config['task_scheduler']['extended_checkpoint']=str(checkpoint);config['task_scheduler']['extended_checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
runtime=SimulationRuntime(config);obs=runtime.reset()
for r in runtime.robots:r.heading=float(f['physical_poses_diagnostic_only'][str(r.robot_id)]['heading'])
obs=runtime._get_observations();adapter=MARVELPolicyAdapter(runtime);adapter.setup();assert adapter._using_policy
scheduler=adapter.scheduler;rec=scheduler._search_recovery;rec.cells.update({(int(c),int(r)):int(v) for c,r,v in f['observed_recovery']['cells']})
if adapter._observed_belief.evidence is not rec:adapter._observed_belief.evidence.cells.update(rec.cells)
for task in runtime.tasks.tasks.values():
 if task.task_type!='relay':task.status='complete'
scheduler.serviced_heat_ids={int(h.heat_id) for h in scheduler.heat_points};scheduler.search_assignments={};scheduler._publish_search_uavs()
rows=[];commands=[];collisions=[];stable=0;start=time.perf_counter()
for mission in range(180):
 actions=adapter.get_actions(obs);probe=getattr(scheduler,'_relay_probe',None)
 commands.append(dict(mission=mission,probe=None if probe is None else dict(uid=int(probe['uid']),anchor=probe['anchor'].tolist()),assignments={str(k):[int(u),a.tolist()] for k,(u,a) in scheduler.relay_assignments.items()},actions=[[np.asarray(p).tolist(),float(h)] for p,h in actions]))
 for tick in range(runtime._mission_step_interval()):
  obs,info=runtime.step(actions);collisions.extend(info['collisions']);stable=stable+1 if info['connectivity_ratio']>=.9 else 0;rows.append(dict(step=runtime.current_step,radio=info['connectivity_ratio'],collisions=info['collisions'],poses={r.robot_id:r.position.tolist() for r in runtime.robots}))
 if mission%10==0:print('RESTORED_RELAY',mission,info['connectivity_ratio'],len(rec.cells),len(scheduler.relay_assignments),None if probe is None else probe['uid'],flush=True)
 if stable>=30 or collisions:break
production_stable=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(passed=stable>=30 and not collisions,saved_pose_fixture=True,natural_acceptance=False,missing_history='MARP graphs rebuilt from current sensing; historical task lifecycle explicitly completed, not exact natural restart',seconds_simulated=len(rows)*runtime.dt,wall_seconds=time.perf_counter()-start,stable_radio_seconds=stable*runtime.dt,final_radio=rows[-1]['radio'],collisions=collisions,probe_events=[e for e in runtime.events if e['type']=='gppo_relay_observed_probe'],trained_checkpoint_sha256=config['task_scheduler']['extended_checkpoint_sha256'],production_source_stable=production_stable,shield_enabled=True,radius=.2,acceptance_claim=False)
(STAGE/'staged_scheduler_saved_pose_physical.json').write_text(json.dumps(result,indent=2));(STAGE/'staged_scheduler_saved_pose_commands.json').write_text(json.dumps(commands));(STAGE/'staged_scheduler_saved_pose_trace.json').write_text(json.dumps(rows));print(json.dumps(result));assert production_stable
