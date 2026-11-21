"""Short natural real-runtime cache equivalence; not efficiency acceptance."""
from pathlib import Path
import sys,json,hashlib,copy,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import benchmark_fair_search as bench
import utils.simulation_runtime as runtime_module
import utils.sensor_models as original_sensor_module
import utils
from gppo_benchmark_domains import load_domain
STAGE=ROOT/'artifacts/extended_exact_sensor_cache_design_20261012_attempt1'
spec=importlib.util.spec_from_file_location('utils.sensor_models_cache_draft',STAGE/'sensor_models_draft.py');draft=importlib.util.module_from_spec(spec);spec.loader.exec_module(draft)
candidate=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15/checkpoint_batch_007.pth';candidate_sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
base_runtime=bench.SimulationRuntime;base_adapter=bench.MARVELPolicyAdapter;base_loader=bench.load_and_validate_scenario;original_factory=runtime_module.create_sensor_model
states=[];commands=[]
def plain(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 raise TypeError(type(value))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=plain).encode()).hexdigest()
class RecordedRuntime(base_runtime):
 def __init__(self,config):
  config['task_scheduler']['extended_checkpoint']=str(candidate);config['task_scheduler']['extended_checkpoint_sha256']=candidate_sha
  super().__init__(config)
 def step(self,actions):
  obs,info=super().step(actions)
  states.append(digest(dict(step=self.current_step,robots=[dict(id=r.robot_id,position=r.position,heading=r.heading,velocity=r.velocity) for r in self.robots],observations=obs,tasks=self.tasks.summary(),connectivity=info['connectivity_ratio'],collisions=info['collisions'])))
  return obs,info
class RecordedAdapter(base_adapter):
 def get_actions(self,observations):
  result=super().get_actions(observations);commands.append(digest(result));return result
bench.SimulationRuntime=RecordedRuntime;bench.MARVELPolicyAdapter=RecordedAdapter
rootout=STAGE/'actual_physics_equivalence_attempt2';rootout.mkdir(exist_ok=False);records=[]
try:
 for n,dynamic,seed in [(30,False,62801),(60,True,62803)]:
  compared=[]
  for cached in [False,True]:
   if n==30 and not cached:
    saved=STAGE/'actual_physics_equivalence_attempt1/30_static_old/equivalence_evidence.json'
    compared.append(json.loads(saved.read_text()));continue
   states.clear();commands.clear()
   def loader(file):return load_domain(base_loader,file,'configs/scenarios/extended_connected_gppo_20261010.yaml',dynamic,.2)
   bench.load_and_validate_scenario=loader
   def factory(config,frame=None):
    selected=copy.deepcopy(config);selected['visible_cells_cache_entries']=128
    return draft.create_sensor_model(selected,frame=frame)
   runtime_module.create_sensor_model=factory if cached else original_factory
   sys.modules['utils.sensor_models']=draft if cached else original_sensor_module
   utils.sensor_models=draft if cached else original_sensor_module
   out=rootout/f'{n}_{"dynamic" if dynamic else "static"}_{"cached" if cached else "old"}';out.mkdir()
   summary=bench.run(out,'gppo',n,seed,12,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=600.)
   events=json.loads((out/'gppo/events.json').read_text());events=[{k:v for k,v in e.items() if k!='timestamp'} for e in events]
   row=dict(uavs=n,dynamic=dynamic,cached=cached,seed=seed,physics_steps=len(states),state_digests=states.copy(),command_digests=commands.copy(),events=events,decisions=json.loads((out/'gppo/decisions.json').read_text()),metric=summary['metric'],collisions=summary['collision_types'],wall_seconds=summary['seconds'])
   (out/'equivalence_evidence.json').write_text(json.dumps(row,indent=2));compared.append(row)
  a,b=compared
  assert a['physics_steps']==b['physics_steps']==120
  for key in ['state_digests','command_digests','events','decisions','metric','collisions']:assert a[key]==b[key],(n,key)
  records.append(dict(uavs=n,domain='dynamic' if dynamic else 'static',seed=seed,physics_steps=120,exact_states=True,exact_commands=True,exact_events=True,exact_actor_graphs=True,collisions=a['collisions'],old_wall_seconds=a['wall_seconds'],cached_wall_seconds=b['wall_seconds']))
finally:
 bench.SimulationRuntime=base_runtime;bench.MARVELPolicyAdapter=base_adapter;bench.load_and_validate_scenario=base_loader;runtime_module.create_sensor_model=original_factory;sys.modules['utils.sensor_models']=original_sensor_module;utils.sensor_models=original_sensor_module
result=dict(passed=True,cases=records,candidate_sha256=candidate_sha,production_installed=False,acceptance_claim=False,scope='Two new independent short12s natural fixtures with exact physical states, observable arrays, commands, public graphs and true events; not complete mission acceptance or whole-runtime speed guarantee.')
(STAGE/'actual_physics_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
