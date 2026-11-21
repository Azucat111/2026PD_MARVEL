"""Controlled same-seed physical parity and cost for staged Safety computation."""
from pathlib import Path
import sys,json,time,importlib.util,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
STAGE=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/performance_stage'
import benchmark_fair_search as bench
import utils.dynamics_models as dynamics
import integrations.gppo.extended_safety_motion as safety
import integrations.gppo.scheduler as scheduler
from utils.simulation_runtime import SimulationRuntime
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['utils','integrations'] for p in (ROOT/d).rglob('*.py')}
old_cls=dynamics.KinematicSimple;old_safety=safety.ObservedMotionSafetyLayer;old_sched=getattr(scheduler,'ObservedMotionSafetyLayer',None);original_step=SimulationRuntime.step
spec=importlib.util.spec_from_file_location('utils.staged_cost_dynamics',STAGE/'dynamics_models.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
spec=importlib.util.spec_from_file_location('integrations.gppo.staged_cost_safety',STAGE/'extended_safety_motion.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
rows=[];traces=[]
for mode in ['original','staged']:
 if mode=='staged':
  dynamics.KinematicSimple=D.KinematicSimple;safety.ObservedMotionSafetyLayer=S.ObservedMotionSafetyLayer
  if old_sched is not None:scheduler.ObservedMotionSafetyLayer=S.ObservedMotionSafetyLayer
 trace=[]
 def capture(runtime,actions):
  obs,info=original_step(runtime,actions)
  trace.append(dict(step=runtime.current_step,poses=[dict(uid=r.robot_id,position=r.position.tolist(),heading=float(r.heading),travel=float(r.travel_distance)) for r in runtime.robots],collision_count=len(info['collisions']),connectivity=float(info['connectivity_ratio'])))
  return obs,info
 SimulationRuntime.step=capture
 source=(ROOT/'scripts/collect_paired_calibration_branch.py').read_text();old="out=ROOT/f'artifacts/extended_paired_calibration_20261014_attempt20/data/seed_{a.seed}_rank_{a.rank}'";assert source.count(old)==1
 source=source.replace(old,f"out=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/performance_stage/physical_{mode}_H6'").replace('a.seed,20,physical_coverage=True','a.seed,6,physical_coverage=True').replace('wall_budget_seconds=240','wall_budget_seconds=180').replace("summary['mission_steps']==20","summary['mission_steps']==6")
 sys.argv=['physical_parity','--seed','63410','--uavs','30','--rank','0','--dynamic']
 namespace=dict(__name__='__main__',__file__=str(ROOT/'scripts/collect_paired_calibration_branch.py'));started=time.perf_counter()
 try:exec(compile(source,str(ROOT/'scripts/collect_paired_calibration_branch.py'),'exec'),namespace)
 finally:SimulationRuntime.step=original_step
 runtime=namespace['current_runtime']
 if mode=='staged':assert type(runtime.high_level_scheduler.safety) is S.ObservedMotionSafetyLayer
 rows.append(dict(mode=mode,seconds=time.perf_counter()-started,metric=namespace['summary']['metric'],full_H6=namespace['summary']['mission_steps']==6));traces.append(trace)
 (STAGE/f'trace_{mode}_H6.json').write_text(json.dumps(trace))
dynamics.KinematicSimple=old_cls;safety.ObservedMotionSafetyLayer=old_safety
if old_sched is not None:scheduler.ObservedMotionSafetyLayer=old_sched
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
exact=traces[0]==traces[1]
result=dict(passed=exact and all(r['full_H6'] for r in rows) and rows[0]['metric']==rows[1]['metric'],physical_ticks=len(traces[0]),uavs=30,exact_positions_headings_travel_radio_collisions=exact,rows=rows,speed_ratio=rows[0]['seconds']/rows[1]['seconds'],production_source_stable=True,acceptance_claim=False,limits='One seed H6 paired trace; not a global speed claim or H300 acceptance')
(STAGE/'physical_parity_cost.json').write_text(json.dumps(result,indent=2));print('STAGED_PHYSICAL_PARITY',json.dumps(result),flush=True);assert result['passed']
