"""Reuse matched static branches; refresh dynamic labels after installed Safety repair."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_public_visibility_preflight_20261015_attempt24'
plan=json.loads((OUT/'plan.json').read_text());frozen=json.loads((OUT/'public_candidates_frozen.json').read_text())
OLD=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20';oldindex=json.loads((OLD/'selected_branches.json').read_text())
index={};jobs=[]
def guard():
 for name,digest in plan['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
def folder(seed,rank):
 p=OUT/'data'/f'seed_{seed}_rank_{rank}'
 complete=[q for q in [p,p.with_name(p.name+'_retry_01')] if (q/'complete.json').exists()]
 assert len(complete)<=1
 return complete[0] if complete else p
for i,seed in enumerate(plan['training_seeds']):
 case=plan['case_cycle'][i%4];state=json.loads((OUT/'data'/f'seed_{seed}'/'public_state.json').read_text())
 candidates=next(r['candidates'] for r in frozen['rankings'] if r['seed']==seed)
 for rank,choice in enumerate(candidates):
  oldrank=next((r for r,o in enumerate(state['top_public_options']) if (o['task'],o['uid'])==(choice['task'],choice['uid'])),None)
  if not case['dynamic'] and oldrank is not None:
   p=OLD/oldindex[f'{seed}:{oldrank}'];old=json.loads((p/'first_decision.json').read_text())
   assert old['graph']==state['graph'] and old['actor_batch']==state['actor_batch']
   assert (old['selected']['task'],old['selected']['uid'])==(choice['task'],choice['uid'])
   index[f'{seed}:{rank}']=dict(folder=str(p.relative_to(ROOT)),reused=True,old_rank=oldrank,reason='No dynamic hazards: new sensing fallback cannot trigger; exact first-state/action match')
  else:jobs.append((seed,case['uavs'],case['dynamic'],rank))
def run(job):
 seed,n,dynamic,rank=job;p=folder(seed,rank)
 if not (p/'complete.json').exists():
  assert not p.with_name(p.name+'_retry_01').exists()
  cmd=[sys.executable,'-u','-B','scripts/collect_public_visibility_preflight_branch.py','--seed',str(seed),'--uavs',str(n),'--rank',str(rank)]
  if dynamic:cmd.append('--dynamic')
  with (OUT/f'branch_{seed}_{rank}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(job,r.returncode)
  p=folder(seed,rank)
 r=json.loads((p/'complete.json').read_text());assert r['bounded_window_complete'] and not r['policy_failures']
 state=json.loads((OUT/'data'/f'seed_{seed}'/'public_state.json').read_text());actual=json.loads((p/'search_decisions.json').read_text())[0]
 assert actual['graph']==state['graph'] and actual['actor_batch']==state['actor_batch'],seed
 choice=next(x['candidates'][rank] for x in frozen['rankings'] if x['seed']==seed)
 assert (actual['selected']['task'],actual['selected']['uid'])==(choice['task'],choice['uid'])
 print('COLLECTED',seed,rank,r['metric']['mean_capped_seconds'],r['collision_types'],flush=True);guard()
 return f'{seed}:{rank}',dict(folder=str(p.relative_to(ROOT)),reused=False)
guard();(OUT/'collection_manifest.json').write_text(json.dumps(dict(reused=len(index),new_branches=len(jobs),source_hashes=plan['source_hashes'],optimizer_updates=0),indent=2))
with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
 for f in as_completed([pool.submit(run,j) for j in jobs]):
  key,value=f.result();index[key]=value
  (OUT/'selected_branches_partial.json').write_text(json.dumps(index,indent=2))
guard();(OUT/'selected_branches.json').write_text(json.dumps(index,indent=2));rows=[]
for seed in plan['training_seeds']:
 records=[json.loads((ROOT/index[f'{seed}:{r}']['folder']/'complete.json').read_text()) for r in range(4)]
 assert records[0]['metric']['detected_targets']==4
 safe=[(r['metric']['mean_capped_seconds'],k) for k,r in enumerate(records) if r['metric']['detected_targets']==4 and not r['collision_types']]
 assert safe,seed
 best,rank=min(safe);candidates=next(r['candidates'] for r in frozen['rankings'] if r['seed']==seed)
 chosen=max(range(4),key=lambda k:(candidates[k]['potential_prior_fraction'],candidates[k]['known_free_prior_fraction'],-candidates[k]['observed_cost'],-k))
 rows.append(dict(seed=seed,baseline=records[0]['metric']['mean_capped_seconds'],best_safe=best,best_rank=rank,means=[r['metric']['mean_capped_seconds'] for r in records],found=[r['metric']['detected_targets'] for r in records],collisions=[r['collision_types'] for r in records],fixed_public_policy_rank=chosen,fixed_public_policy_mean=records[chosen]['metric']['mean_capped_seconds']))
oracle=sum(r['baseline'] for r in rows)/sum(r['best_safe'] for r in rows)-1
fixed=sum(r['baseline'] for r in rows)/sum(r['fixed_public_policy_mean'] for r in rows)-1
result=dict(training_only=True,rows=rows,oracle_safe_gain=oracle,fixed_public_policy_training_gain=fixed,eligible_for_low_cost_learning=oracle>=.15,acceptance_claim=False,goal_complete=False,expensive_training_allowed=False)
(OUT/'scope_ceiling_complete.json').write_text(json.dumps(result,indent=2));print('PUBLIC_SCOPE_CEILING',json.dumps(result),flush=True)
