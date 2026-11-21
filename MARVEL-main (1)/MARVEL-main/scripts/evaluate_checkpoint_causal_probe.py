"""Evaluation-only single arm on a frozen common suite. Never constructs an optimizer."""
from pathlib import Path
import sys,json,hashlib,argparse,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
import torch,numpy as np
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from integrations.gppo.opportunity_features import opportunity_batch,public_search_context
from gppo_benchmark_domains import load_domain
from gppo_joint_completion_metric import joint_completion_metric
from gppo_training_records import actor_input_record
from gppo_motion_capture import capture_motion_frame,append_motion_frame,json_value
OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
plan=json.loads((OUT/'plan.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--case',required=True,type=int);p.add_argument('--arm',required=True,choices=plan['arms']);p.add_argument('--pilot',action='store_true');p.add_argument('--alternative',required=True,choices=list(plan['models']));p.add_argument('--retry',default='');a=p.parse_args()
assert not plan['training_permitted']
causal_plan=json.loads((OUT/'causal_plan.json').read_text());intervention=next(j for j in causal_plan['jobs'] if j['case_index']==a.case and j['original_arm']==a.arm and j['alternative_arm']==a.alternative)
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==causal_plan['causal_runner_sha256']
case=plan['cases'][a.case];seed=case['seed'];n=case['uavs'];dynamic=case['dynamic']
assert seed not in plan['reserved_seeds_untouched']
model=plan['models']['round19' if a.arm=='rule' else a.arm]
candidate=ROOT/model['path']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 for f,h in plan['source_hashes'].items():assert sha(ROOT/f)==h,f
 for m in plan['models'].values():assert sha(ROOT/m['path'])==m['sha256']
 assert sha(Path(plan['frozen_checkpoint']))==plan['frozen_sha256']
guard()
base=OUT/'causal_probes'/f'case_{a.case:02d}_{a.arm}_to_{a.alternative}'
out=base if not a.retry else base.with_name(base.name+'_'+a.retry)
out.mkdir(parents=True,exist_ok=False)
(out/'run_started.json').write_text(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=a.arm,case=case,pilot=a.pilot,checkpoint_sha256=model['sha256'],optimizer_updates=0),indent=2))
base_runtime=bench.SimulationRuntime;base_adapter=bench.MARVELPolicyAdapter;base_loader=bench.load_and_validate_scenario;base_rule=bench.choose_rule_assignment
runtime=None;joint=[];records=[];motion_frames=0;causal_log=[]
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.adapter import GPPOAssignment
alternate_spec=plan['models'][a.alternative]
with torch.random.fork_rng(devices=[]):alternate_model=GPPOActorCritic(edge_feature_dim=alternate_spec['edge_dim'])
alternate_model.load_state_dict(torch.load(ROOT/alternate_spec['path'],map_location='cpu',weights_only=False)['model']);alternate_model.eval()
class Runtime(base_runtime):
 def __init__(self,config):
  global runtime
  config['task_scheduler']['extended_checkpoint']=str(candidate)
  config['task_scheduler']['extended_checkpoint_sha256']=model['sha256']
  super().__init__(config);runtime=self
  assert self.dt==.1
  assert config['scenario']['duration']['max_steps']>=3000
 def step(self,actions):
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
 row['actual_model_edge_dim']=model['edge_dim'] if a.arm!='rule' else None
 row['raw_edge_seconds']=graph.edge_features.tolist()
 row['public_context']=observer.public_search_context
 row['raw_uav_features']=graph.uav_features.tolist();row['raw_task_features']=graph.task_features.tolist()
 row['step']=int(runtime.current_step)
 intervened=bool(causal_log and causal_log[-1]['record_index']==len(records))
 row['allocation_policy_source']=a.alternative if intervened else a.arm
 if intervened:row['actual_model_edge_dim']=alternate_spec['edge_dim']
 records.append(row)
def rule(graph,deterministic=True):
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
   choice=original(graph,deterministic)
   if choice is not None and len(records)==intervention['record_index']:
    assert int(self.runtime.current_step)==intervention['step']
    assert [int(choice.subtask_id),int(choice.uav_id)]==intervention['baseline_task_uid']
    observer=inference.heading_observations;observer.public_search_context=public_search_context(self.scheduler)
    b=opportunity_batch(graph,observer,'cpu')
    digest=hashlib.sha256(json.dumps({k:v.tolist() for k,v in b.items()},sort_keys=True).encode()).hexdigest()
    assert digest==intervention['public_state_sha256'],'Causal prefix is not the recorded identical public state'
    bb=dict(b);bb['edge_features']=bb['edge_features'][...,:alternate_spec['edge_dim']].contiguous()
    with torch.no_grad():o=alternate_model(**bb)
    action=int(o['flat_logits'][0].argmax());ti,ui=divmod(action,graph.num_uavs)
    assert not graph.action_mask[ti,ui]
    assert [int(graph.subtasks[ti].subtask_id),int(graph.uav_states[ui].uav_id)]==intervention['alternative_task_uid']
    causal_log.append(dict(step=int(self.runtime.current_step),record_index=len(records),original=dict(task=choice.subtask_id,uid=choice.uav_id),alternative=dict(task=int(graph.subtasks[ti].subtask_id),uid=int(graph.uav_states[ui].uav_id)),public_state_sha256=digest))
    choice=GPPOAssignment(action,ti,ui,int(graph.subtasks[ti].subtask_id),int(graph.uav_states[ui].uav_id),float(o['value'][0]))
   record(graph,choice,inference)
   return choice
  inference.assign=assign
  return result
 def get_actions(self,observations):
  global motion_frames
  actions=super().get_actions(observations);s=self.scheduler
  # Full public executor snapshots while Search is outstanding; compact physical
  # diagnostics afterwards. None of these records feed the policy.
  found=sum(e['type']=='target_found' for e in self.runtime.events)
  if found<self.runtime.search_scenario.target_count:
   frame=capture_motion_frame(self.runtime,s,observations,actions)
  else:
   frame=dict(step=int(self.runtime.current_step),time=float(self.runtime.current_step*self.runtime.dt),public_search_assignments=dict(s.search_assignments),public_relay_assignments={int(k):dict(helper=int(v[0]),anchor=v[1]) for k,v in s.relay_assignments.items()},physical_poses_diagnostic_only={int(r.robot_id):dict(position=r.position,heading=float(r.heading)) for r in self.runtime.robots})
  append_motion_frame(out/('rule' if a.arm=='rule' else 'gppo')/'mission_frames.jsonl',frame)
  motion_frames+=1
  return actions
bench.SimulationRuntime=Runtime;bench.MARVELPolicyAdapter=Adapter
bench.load_and_validate_scenario=lambda f:load_domain(base_loader,f,plan['scenario'],dynamic,plan['warning_rate_mps'])
bench.choose_rule_assignment=rule
horizon=2 if a.pilot else plan['horizon_seconds']
source=(ROOT/'scripts/benchmark_fair_search.py').read_text();source=source[:source.index("if __name__=='__main__':")]
needle="if (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break"
replacement="if detection_efficiency(runtime.events,runtime.search_scenario.target_count,runtime.dt,steps*interval*runtime.dt)['detected_targets']==runtime.search_scenario.target_count or (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break"
assert source.count(needle)==1
ns=dict(bench.__dict__);exec(compile(source.replace(needle,replacement),str(ROOT/'scripts/benchmark_fair_search.py'),'exec'),ns)
ns.update(SimulationRuntime=Runtime,MARVELPolicyAdapter=Adapter,load_and_validate_scenario=bench.load_and_validate_scenario,choose_rule_assignment=bench.choose_rule_assignment)
summary=ns['run'](out,'rule' if a.arm=='rule' else 'gppo',n,seed,horizon,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=plan['wall_budget_seconds_per_arm'])
folder=out/summary['arm']
assert summary['metric']['detected_targets']==4,'No complete actual-Find outcome in bounded causal probe'
assert len(causal_log)==1,'Exactly one allocator intervention required'
assert not summary['policy_execution_failures']
summary['full_window_complete']=summary['mission_steps']==horizon
summary['secondary_joint_metric']=dict(scored=False,joint_success=False,reason='Early allFind causal probe; no full-window joint claim')
summary['causal_intervention']=causal_log
(out/'intervention.json').write_text(json.dumps(causal_log,indent=2))
(folder/'summary.json').write_text(json.dumps(summary,indent=2))
(folder/'physical_joint_trace.json').write_text(json.dumps(joint,indent=2))
(folder/'actor_inputs_9.json').write_text(json.dumps(records,indent=2))
(folder/'private_target_identity_diagnostic_only.json').write_text(json.dumps(dict(targets=runtime.target_detector.hidden_targets),default=json_value,indent=2))
guard()
result=dict(arm=a.arm,case=case,pilot=a.pilot,summary=summary,checkpoint_sha256=model['sha256'],motion_frames=motion_frames,source_stable=True,optimizer_updates=0,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_complete=False)
(out/'complete.json').write_text(json.dumps(result,indent=2))
print('CAUSAL_PROBE_COMPLETE',json.dumps(dict(arm=a.arm,case=a.case,seed=seed,uavs=n,dynamic=dynamic,mean_find=summary['metric']['mean_capped_seconds'],joint=summary['secondary_joint_metric'],collisions=summary['collision_types'],wall_seconds=summary['seconds'])),flush=True)
