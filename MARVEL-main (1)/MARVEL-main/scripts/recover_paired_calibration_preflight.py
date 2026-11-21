from pathlib import Path
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20'
plan=json.loads((OUT/'recovery_plan_20261015.json').read_text());index=json.loads((OUT/'selected_branches.json').read_text());initial=json.loads((OUT/'plan.json').read_text())
def state(phase,**kw):(OUT/'recovery_state.json').write_text(json.dumps(dict(phase=phase,goal_complete=False,long_training_started=False,**kw),indent=2))
def guard():
 for f,h in plan['new_source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
 for f,h in initial['protected_weights'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
guard();state('RECOVERING_ONLY_FIVE_UNFINISHED_BRANCHES',reuse=43)
for job in plan['retry_only']:
 seed=job['seed'];rank=job['rank'];name=f'seed_{seed}_rank_{rank}_retry_01';folder=OUT/'data'/name
 if not (folder/'complete.json').exists():
  assert not folder.exists(),f'Preserved partial recovery requires explicit handling: {folder}'
  cmd=[sys.executable,'-u','-B','scripts/retry_paired_calibration_branch.py','--seed',str(seed),'--uavs',str(job['uavs']),'--rank',str(rank)]
  if job['dynamic']:cmd.append('--dynamic')
  with (OUT/f'{name}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:state('RECOVERY_FAILED',job=job,exit_code=r.returncode);raise SystemExit(r.returncode)
 proof=json.loads((folder/'complete.json').read_text());assert proof['full_short_window'] and not proof['policy_failures'],job
 previous=json.loads((OUT/index[f'{seed}:{rank}']/'first_decision.json').read_text());fresh=json.loads((folder/'first_decision.json').read_text())
 assert fresh['actor_batch']==previous['actor_batch'] and fresh['graph']==previous['graph'] and fresh['selected']==previous['selected'],job
 index[f'{seed}:{rank}']='data/'+name;(OUT/'selected_branches.json').write_text(json.dumps(index,indent=2));guard()
 print('RECOVERED',seed,rank,flush=True)
for seed in initial['training_seeds']+initial['validation_seeds']:
 first=None;init=None;cfg=None
 for rank in range(4):
  p=OUT/index[f'{seed}:{rank}'];r=json.loads((p/'complete.json').read_text());f=json.loads((p/'first_decision.json').read_text());a=json.loads((p/'rule/initial.json').read_text());c=json.loads((p/'rule/config.json').read_text())
  assert r['full_short_window'] and not r['policy_failures']
  if first is None:first=f;init=a;cfg=c
  assert f['actor_batch']==first['actor_batch'] and f['graph']==first['graph'] and a==init and c==cfg,seed
guard();(OUT/'collection_complete_recovered.json').write_text(json.dumps(dict(complete=True,branches=48,original_complete_reused=43,recovered=5,matched_public_inputs=True,matched_initial_conditions=True,semantics_preserving_computation_revision=True,proof='performance_stage/physical_parity_cost_v2.json',acceptance_claim=False),indent=2))
state('ONE_REGISTERED_FIT')
with (OUT/'recovered_fit_stdout.log').open('a') as log:r=subprocess.run([sys.executable,'-u','-B','scripts/fit_recovered_paired_calibration_once.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:state('FIT_FAILED',exit_code=r.returncode);raise SystemExit(r.returncode)
result=json.loads((OUT/'validation_complete.json').read_text());guard();state('SHORT_VALIDATION_PASSED' if result['eligible_for_h300_development'] else 'SHORT_VALIDATION_REJECTED',gain=result['pooled_gain'],candidate_sha256=result['candidate_sha256'])
