"""Bounded training-only action ceiling before any learning."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_second_search_preflight_20261015_attempt21';plan=json.loads((OUT/'plan.json').read_text())
def folder(seed,rank):
 base=OUT/'data'/f'seed_{seed}_rank_{rank}'
 complete=[p for p in [base,base.with_name(base.name+'_retry_01')] if (p/'complete.json').exists()]
 assert len(complete)<=1
 return complete[0] if complete else base
def run(job):
 seed,n,dynamic,rank=job;p=folder(seed,rank)
 if (p/'complete.json').exists():return json.loads((p/'complete.json').read_text())
 assert not p.with_name(p.name+'_retry_01').exists(),f'Incomplete retry preserved: {p}'
 cmd=[sys.executable,'-u','-B','scripts/collect_second_search_preflight_branch.py','--seed',str(seed),'--uavs',str(n),'--rank',str(rank)]
 if dynamic:cmd.append('--dynamic')
 with (OUT/f'seed_{seed}_rank_{rank}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(job,r.returncode)
 p=folder(seed,rank)
 result=json.loads((p/'complete.json').read_text());assert result['bounded_window_complete'] and not result['policy_failures'],job
 print('COLLECTED',seed,rank,result['metric']['mean_capped_seconds'],flush=True)
 return result
jobs=[]
for i,seed in enumerate(plan['training_seeds']):
 c=plan['case_cycle'][i%4]
 for rank in plan['candidate_ranks']:jobs.append((seed,c['uavs'],c['dynamic'],rank))
with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):future.result()
old=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20';index=json.loads((old/'selected_branches.json').read_text());rows=[]
for seed in plan['training_seeds']:
 states=[];metrics=[]
 for rank in plan['candidate_ranks']:
  p=folder(seed,rank);states.append(json.loads((p/'search_decisions.json').read_text()));r=json.loads((p/'complete.json').read_text());metrics.append(r)
 for state in states[1:]:
  assert state[0]==states[0][0]
  for key in ['time','teacher','top_public_options','actor_batch','graph']:assert state[1][key]==states[0][1][key],(seed,key)
 baseline=json.loads((old/index[f'{seed}:0']/'complete.json').read_text())
 assert baseline['metric']['detected_targets']==4 and not baseline['collision_types']
 legal=[(r['metric']['mean_capped_seconds'],rank) for rank,r in enumerate(metrics) if r['metric']['detected_targets']==4 and not r['collision_types']]
 assert legal,(seed,'no exact safe candidate')
 best,rank=min(legal)
 rows.append(dict(seed=seed,baseline_mean=baseline['metric']['mean_capped_seconds'],best_safe_second_rank=rank,best_mean=best,means=[r['metric']['mean_capped_seconds'] for r in metrics],found=[r['metric']['detected_targets'] for r in metrics]))
gain=sum(r['baseline_mean'] for r in rows)/sum(r['best_mean'] for r in rows)-1
result=dict(training_only=True,oracle_only=True,deployable_policy=False,acceptance_claim=False,goal_complete=False,optimizer_updates=0,rows=rows,oracle_pooled_gain=gain,eligible_for_low_cost_fit=gain>=.15)
(OUT/'scope_ceiling_complete.json').write_text(json.dumps(result,indent=2));print('SCOPE_CEILING',json.dumps(result),flush=True)
