"""Fixed diagnostic cases only; no independent acceptance and no model selection. Never constructs an optimizer."""
from pathlib import Path
import sys,json,hashlib,argparse,datetime
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
import torch,numpy as np
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from integrations.gppo.opportunity_features import opportunity_batch,public_search_context
from gppo_benchmark_domains import load_domain
from gppo_joint_completion_metric import joint_completion_metric
from gppo_training_records import actor_input_record
from gppo_motion_capture import append_motion_frame,json_value
OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/'plan.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--case',required=True,type=int);p.add_argument('--arm',required=True,choices=plan['arms']);p.add_argument('--pilot',action='store_true');p.add_argument('--retry',default='');p.add_argument('--mode',choices=['original'],required=True);a=p.parse_args()
assert not plan['training_permitted']
case=plan['cases'][a.case];seed=case['seed'];n=case['uavs'];dynamic=case['dynamic']
assert (seed,n,dynamic) in ((1880021, 30, False), (1280022, 30, True), (1880024, 60, True))
assert plan['diagnostic_training_cases'] is True and not plan['independent_acceptance']
assert not plan['model_selection_permitted']
assert plan['horizon_seconds']==300 and plan['physics_steps']==3000
model=plan['models'].get(a.arm,plan['models']['round8'])
candidate=ROOT/model['path']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 for f,h in plan['source_hashes'].items():assert sha(ROOT/f)==h,f
 for m in plan['models'].values():assert sha(ROOT/m['path'])==m['sha256']
 assert sha(Path(plan['frozen_checkpoint']))==plan['frozen_sha256']
 assert sha(ROOT/plan['update_proof_path'])==plan['update_proof_sha256']
 assert sha(OUT/'plan.json')==(OUT/'protocol_sha256.txt').read_text().strip()
guard()
base=OUT/('pilots' if a.pilot else 'data')/f"case_{a.case:02d}_{seed}_{n}_{'dynamic' if dynamic else 'static'}"/(a.arm+'_'+a.mode)
out=base if not a.retry else base.with_name(base.name+'_'+a.retry)
out.mkdir(parents=True,exist_ok=False)
(out/'run_started.json').write_text(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=a.arm,case=case,pilot=a.pilot,checkpoint_sha256=model['sha256'],optimizer_updates=0),indent=2))
base_runtime=bench.SimulationRuntime;base_adapter=bench.MARVELPolicyAdapter;base_loader=bench.load_and_validate_scenario;base_rule=bench.choose_rule_assignment
import time
DEADLINE=datetime.datetime.fromisoformat("2026-11-11T06:20:00+00:00").timestamp()
def check_deadline():
 if time.time()>=DEADLINE:raise TimeoutError("Fixed diagnostic runtime deadline")
check_deadline()
runtime=None;joint=[];records=[];motion_frames=0;interventions=[]
class Runtime(base_runtime):
 def __init__(self,config):
  global runtime
  config['task_scheduler']['extended_checkpoint']=str(candidate)
  config['task_scheduler']['extended_checkpoint_sha256']=model['sha256']
  super().__init__(config);runtime=self
  assert self.dt==.1
  assert config['scenario']['duration']['max_steps']>=3000
 def step(self,actions):
  check_deadline()
  obs,info=super().step(actions);s=getattr(self,'high_level_scheduler',None)
  if s is not None:joint.append(dict(time=self.current_step*self.dt,connectivity=float(info['connectivity_ratio']),nonrelay_complete=self.tasks.all_complete(),safety_active=sorted(s.safety.active_uav_ids),safety_records=len(s.safety.records),escapes=s.safety.successful_escapes,no_fly_violations=s.safety.no_fly_violations,collisions=len(info['collisions'])))
  return obs,info
def record(graph,choice,adapter):
 if choice is None:return
 observer=adapter.heading_observations
 observer.public_search_context=public_search_context(runtime.high_level_scheduler)
 batch=opportunity_batch(graph,observer,'cpu')
 if a.arm!='rule':
  real=adapter.last_batch
  for k,v in real.items():
   expected=batch[k][...,:model['edge_dim']] if k=='edge_features' else batch[k]
   assert torch.equal(v.cpu(),expected),('Actual input capture mismatch',k)
 row=actor_input_record(batch,time=graph.current_time,action=choice.flat_action,uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks],value=choice.value)
 row['source_policy']=getattr(adapter,'last_role_policy',a.arm)
 row['actual_model_edge_dim']=model['edge_dim'] if a.arm!='rule' else None
 row['raw_edge_seconds']=graph.edge_features.tolist()
 row['public_context']=observer.public_search_context
 row['raw_uav_features']=graph.uav_features.tolist();row['raw_task_features']=graph.task_features.tolist()
 row['step']=int(runtime.current_step)
 records.append(row)
def rule(graph,deterministic=True):
 assert deterministic,'Diagnostic evaluation must use deterministic actions'
 choice=base_rule(graph,deterministic)
 if a.arm=='rule':record(graph,choice,runtime.high_level_scheduler.allocator.adapter)
 return choice
class Adapter(base_adapter):
 def __init__(self,*args,**kwargs):
  with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
   super().__init__(*args,**kwargs)
 def setup(self):
  with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
   result=super().setup()
  inference=self.scheduler.allocator.adapter
  assert isinstance(inference,ExtendedGPPOInferenceAdapter)
  assert inference.candidate_sha256==model['sha256']
  assert inference.model.ahgnn.edge_input.weight.shape[1]==model['edge_dim']
  assert not inference.model.training

  original=inference.assign
  def assign(graph,deterministic=True):
   assert deterministic,'Diagnostic evaluation must use deterministic actions'
   choice=original(graph,deterministic)
   record(graph,choice,inference)
   return choice
  inference.assign=assign
  return result
 def get_actions(self,observations):
  global motion_frames
  actions=super().get_actions(observations);s=self.scheduler
  frame=dict(step=int(self.runtime.current_step),time=float(self.runtime.current_step*self.runtime.dt),
   public_search_assignments=dict(s.search_assignments),
   public_relay_assignments={int(k):dict(helper=int(v[0]),anchor=v[1]) for k,v in s.relay_assignments.items()},
   safety_active=sorted(s.safety.active_uav_ids),
   physical_poses_diagnostic_only={int(r.robot_id):dict(position=r.position,heading=float(r.heading)) for r in self.runtime.robots})
  append_motion_frame(out/('rule' if a.arm=='rule' else 'gppo')/'mission_frames.jsonl',frame)
  motion_frames+=1
  return actions
bench.SimulationRuntime=Runtime;bench.MARVELPolicyAdapter=Adapter
bench.load_and_validate_scenario=lambda f:load_domain(base_loader,f,plan['scenario'],dynamic,plan['warning_rate_mps'])
bench.choose_rule_assignment=rule
horizon=plan['horizon_seconds']
summary=bench.run(out,'rule' if a.arm=='rule' else 'gppo',n,seed,horizon,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=plan['wall_budget_seconds_per_arm'])
folder=out/summary['arm']
assert summary['mission_steps']==horizon,'Incomplete window; preserve partial and do not score'
assert len(joint)==horizon*10
assert not summary['policy_execution_failures']
summary['full_window_complete']=True
summary['diagnostic_training_cases']=True
summary['independent_diagnostic']=False
summary['independent_acceptance']=False
summary['model_selection_permitted']=False
summary['causal_mode']=a.mode
summary['handoff_interventions']=interventions
summary['secondary_joint_metric']=joint_completion_metric(joint,horizon,physics_dt=runtime.dt)
(folder/'summary.json').write_text(json.dumps(summary,indent=2))
(folder/'physical_joint_trace.json').write_text(json.dumps(joint,indent=2))
(folder/'actor_inputs_9.json').write_text(json.dumps(records,indent=2))
(folder/'private_target_identity_diagnostic_only.json').write_text(json.dumps(dict(targets=runtime.target_detector.hidden_targets),default=json_value,indent=2))
guard()
result=dict(arm=a.arm,case=case,pilot=a.pilot,summary=summary,checkpoint_sha256=model['sha256'],motion_frames=motion_frames,source_stable=True,optimizer_updates=0,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_complete=False,mode=a.mode,diagnostic_training_cases=True,independent_diagnostic=False,independent_acceptance=False,model_selection_permitted=False)
(out/'complete.json').write_text(json.dumps(result,indent=2))
print('COMMON_ARM_COMPLETE',json.dumps(dict(arm=a.arm,case=a.case,seed=seed,uavs=n,dynamic=dynamic,mean_find=summary['metric']['mean_capped_seconds'],joint=summary['secondary_joint_metric'],collisions=summary['collision_types'],wall_seconds=summary['seconds'])),flush=True)
