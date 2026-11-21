"""Controlled still-pose whole runtime step equivalence/performance, not task acceptance."""
from pathlib import Path
import sys,json,time,random,hashlib,copy,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,utils
import utils.sensor_models as original_sensor
import utils.simulation_runtime as runtime_module
from utils.simulation_runtime import SimulationRuntime
STAGE=ROOT/'artifacts/extended_exact_sensor_cache_design_20261012_attempt1'
spec=importlib.util.spec_from_file_location('utils.sensor_models_cache_draft',STAGE/'sensor_models_draft.py');draft=importlib.util.module_from_spec(spec);spec.loader.exec_module(draft)
config=json.loads((STAGE/'actual_physics_equivalence_attempt2/60_dynamic_old/gppo/config.json').read_text());config['dynamic_obstacles']=[];config['scenario']['random_seed']=62805
original_factory=runtime_module.create_sensor_model

def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v))

def run(cached):
 random.seed(62805);np.random.seed(62805)
 def factory(cfg,frame=None):
  cfg=copy.deepcopy(cfg);cfg['visible_cells_cache_entries']=128;return draft.create_sensor_model(cfg,frame=frame)
 runtime_module.create_sensor_model=factory if cached else original_factory
 sys.modules['utils.sensor_models']=draft if cached else original_sensor;utils.sensor_models=draft if cached else original_sensor
 runtime=SimulationRuntime(copy.deepcopy(config));runtime.reset();actions=[(r.position.copy(),r.heading) for r in runtime.robots];digests=[];collisions=0;start=time.perf_counter()
 for step in range(20):
  obs,info=runtime.step(actions);collisions+=len(info['collisions']);payload=dict(obs=obs,poses=[(r.position,r.heading) for r in runtime.robots],step=runtime.current_step,tasks=runtime.tasks.summary(),radio=info['connectivity_ratio'],collisions=info['collisions']);digests.append(hashlib.sha256(json.dumps(payload,sort_keys=True,default=plain).encode()).hexdigest())
 seconds=time.perf_counter()-start
 sensors=list(runtime.sensor_by_type.values());return dict(cached=cached,seconds=seconds,digests=digests,collisions=collisions,hits=sum(getattr(s,'_visible_cache_hits',0) for s in sensors),misses=sum(getattr(s,'_visible_cache_misses',0) for s in sensors))
rows=[]
try:
 for trial in range(3):
  pair=[run(c) for c in ([False,True] if trial%2==0 else [True,False])];old=next(p for p in pair if not p['cached']);new=next(p for p in pair if p['cached']);assert old['digests']==new['digests'];assert old['collisions']==new['collisions']==0;rows.append(dict(trial=trial,old=old,cached=new,speedup=old['seconds']/new['seconds']))
finally:
 runtime_module.create_sensor_model=original_factory;sys.modules['utils.sensor_models']=original_sensor;utils.sensor_models=original_sensor
result=dict(passed=True,uavs=60,seed=62805,physics_steps_per_run=20,trials=3,all_exact_physical_and_observation_hashes=True,zero_collisions=True,rows=rows,median_controlled_still_step_speedup=float(np.median([r['speedup'] for r in rows])),production_installed=False,acceptance_claim=False,scope='Controlled legal stationary actions only, real dynamics/collision/radio/sensing; does not demonstrate full mission wall-time or scheduling efficiency gain.')
(STAGE/'controlled_still_runtime_equality_and_cpu.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
