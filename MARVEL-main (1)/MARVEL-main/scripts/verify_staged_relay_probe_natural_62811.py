"""Fresh natural short development of staged Relay recovery; no acceptance claim."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));STAGE=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt1'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt17.json').read_text());hashes={**plan['source_hashes'],**plan['pipeline_hashes']};assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
for name,file in [('integrations.gppo.relay_frontier','relay_frontier.py'),('integrations.gppo.scheduler','scheduler_draft.py'),('utils.policy_adapter','policy_adapter_draft.py')]:
 spec=importlib.util.spec_from_file_location(name,STAGE/file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);parent,attr=name.rsplit('.',1);setattr(sys.modules[parent],attr,module)
 if name=='integrations.gppo.scheduler':sys.modules['integrations.gppo'].GPPOTaskScheduler=module.GPPOTaskScheduler
 if name=='utils.policy_adapter':module._MARVEL_ROOT=ROOT
import benchmark_fair_search as bench
from gppo_benchmark_domains import load_domain
from gppo_joint_completion_metric import joint_completion_metric
original_loader=bench.load_and_validate_scenario;bench.load_and_validate_scenario=lambda file:load_domain(original_loader,file,plan['scenario'],False,.2)
base=bench.SimulationRuntime;rows=[];candidate=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17/checkpoint_batch_015.pth';candidate_sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
class StagedRuntime(base):
 def __init__(self,config):
  config['task_scheduler']['extended_checkpoint']=str(candidate);config['task_scheduler']['extended_checkpoint_sha256']=candidate_sha;super().__init__(config)
 def step(self,actions):
  obs,info=super().step(actions);s=self.high_level_scheduler
  assert s.__class__ is sys.modules['integrations.gppo.scheduler'].GPPOTaskScheduler
  rows.append(dict(time=self.current_step*self.dt,connectivity=info['connectivity_ratio'],nonrelay_complete=self.tasks.all_complete(),safety_active=sorted(s.safety.active_uav_ids),safety_records=len(s.safety.records),escapes=s.safety.successful_escapes,no_fly_violations=s.safety.no_fly_violations,collisions=len(info['collisions'])))
  return obs,info
bench.SimulationRuntime=StagedRuntime;out=STAGE/'natural_30_static_62811';out.mkdir(exist_ok=False)
summary=bench.run(out,'gppo',30,62811,150,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=1800.)
result=dict(natural_from_reset=True,development_only=True,checkpoint_sha256=candidate_sha,summary=summary,secondary_joint_metric=joint_completion_metric(rows,150),production_source_stable=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),staged_source_hashes={p:hashlib.sha256((STAGE/p).read_bytes()).hexdigest() for p in ['relay_frontier.py','scheduler_draft.py','policy_adapter_draft.py']},acceptance_claim=False)
(STAGE/'natural_30_static_62811.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['production_source_stable']
