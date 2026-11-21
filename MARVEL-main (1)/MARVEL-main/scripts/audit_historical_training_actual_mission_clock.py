"""Offline audit of actual historical mission-clock headings; no environment rerun."""
from pathlib import Path
import sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.heading_features import HeadingObservations
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt11'
rows=[]
for i in range(16):
 ep=folder/f'episode_{i:03d}'
 records=json.loads((ep/'actor_inputs.json').read_text())
 decisions=json.loads((ep/'gppo/decisions.json').read_text())
 config=json.loads((ep/'gppo/config.json').read_text())
 interval=int(config['_gppo_profile']['high_level_interval_steps'])
 with (ep/'gppo/mission_frames.jsonl').open() as probe:
  nonzero=next(json.loads(line) for line in probe if json.loads(line)['step']>0)
 dt=float(nonzero['time'])/int(nonzero['step']);assert dt>0
 observer=HeadingObservations(.05); snapshots={};steps=[]
 maxstep=max(int(round(r['time']/dt)) for r in records)
 with (ep/'gppo/mission_frames.jsonl').open() as f:
  for line in f:
   frame=json.loads(line)
   step=int(frame['step']);assert abs(frame['time']-step*dt)<1e-9
   if step>maxstep:break
   poses=frame['physical_poses_diagnostic_only']
   observer.observe({int(uid):{'heading':p['heading']} for uid,p in poses.items()})
   snapshots[step]={uid:observer.heading(uid) for uid in observer.current};steps.append(step)
 assert steps==list(range(0,maxstep+1,interval)),steps
 errs=[]
 for record in records:
  step=int(round(record['time']/dt));heads=snapshots[step]
  matches=[d for d in decisions if d['step']==step and d['uav_ids']==record['uav_ids'] and d['task_ids']==record['task_ids'] and d['chosen']==record['chosen']]
  assert len(matches)==1,(i,step,len(matches))
  d=matches[0];actual=np.asarray(record['batch']['edge_features'],dtype=np.float32)[0,:,:,3:];expected=np.zeros_like(actual)
  for t,task in enumerate(d['raw_task_features']):
   for u,(uid,uf) in enumerate(zip(d['uav_ids'],d['raw_uav_features'])):
    bearing=math.atan2(task[11]-uf[10],task[10]-uf[9]);relative=bearing-math.radians(heads[uid])
    expected[t,u]=[math.cos(relative),math.sin(relative),1.]
  errs.append(float(np.max(np.abs(actual-expected))))
 assert max(errs)<1e-4,(i,max(errs))
 rows.append(dict(episode=i,records=len(records),mission_interval_physics_ticks=interval,physics_dt=dt,actual_observation_interval_seconds=interval*dt,max_heading_feature_error=max(errs)))
result=dict(passed=True,episodes=16,records=sum(r['records'] for r in rows),rows=rows,max_heading_feature_error=max(r['max_heading_feature_error'] for r in rows),conclusion='Actual historical benchmark get_actions cadence already1s: previous0.1s claim applies hypothetical every-physics-tick adapter calls, not these saved training episodes.',limitations='Heading history reconstructed from saved diagnostic physical self-heading; public graph positions stored float32, not original float64; raw sensor snapshots for unassigned UAVs missing. Source sensor identity and frame cadence checked separately. No hidden obstacle or target state supplied to actor.',production_changed=False,acceptance_claim=False)
with (folder/'actual_mission_clock_reassessment_v2.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
