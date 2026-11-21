"""Two predeclared continuous teachers, common baselines reused; no optimizer updates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_expected_find_preflight_20261015_attempt27'
plan=json.loads((OUT/'plan.json').read_text());BASE=ROOT/'artifacts/extended_public_visibility_preflight_20261015_attempt24';index=json.loads((BASE/'selected_branches.json').read_text())
def guard():
 for n,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
def run(job):
 seed,n,dynamic,mode=job;p=OUT/'data'/f'seed_{seed}_{mode}'
 if not (p/'complete.json').exists():
  assert not p.exists(),f'Incomplete branch preserved: {p}'
  cmd=[sys.executable,'-u','-B','scripts/evaluate_expected_find_teacher_branch.py','--seed',str(seed),'--uavs',str(n),'--mode',mode]
  if dynamic:cmd.append('--dynamic')
  with (OUT/f'branch_{seed}_{mode}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(job,r.returncode)
 r=json.loads((p/'complete.json').read_text());assert r['bounded_window_complete'] and not r['policy_failures']
 base=ROOT/index[f'{seed}:0']['folder']
 for f in ['config.json','initial.json']:assert json.loads((base/'rule'/f).read_text())==json.loads((p/'rule'/f).read_text()),(seed,f)
 observed=json.loads((p/'teacher_decisions.json').read_text())[0]
 state=json.loads((BASE/'data'/f'seed_{seed}'/'public_state.json').read_text())
 assert observed['actor_batch']==state['actor_batch'],seed
 guard();print('TEACHER_PAIRED',seed,mode,r['metric']['mean_capped_seconds'],r['collision_types'],flush=True)
jobs=[]
for i,seed in enumerate(plan['training_seeds']):
 c=plan['case_cycle'][i%4]
 for mode in plan['teachers']:jobs.append((seed,c['uavs'],c['dynamic'],mode))
guard()
with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
 for f in as_completed([pool.submit(run,j) for j in jobs]):f.result()
results=[]
for mode in plan['teachers']:
 rows=[]
 for seed in plan['training_seeds']:
  b=json.loads((ROOT/index[f'{seed}:0']['folder']/'complete.json').read_text());c=json.loads((OUT/'data'/f'seed_{seed}_{mode}'/'complete.json').read_text())
  rows.append(dict(seed=seed,baseline=b['metric'],candidate=c['metric'],collisions=c['collision_types'],overrides=c['overrides']))
 gain=sum(r['baseline']['mean_capped_seconds'] for r in rows)/sum(r['candidate']['mean_capped_seconds'] for r in rows)-1
 eligible=gain>=.15 and all(r['candidate']['detected_targets']==4 and not r['collisions'] for r in rows)
 results.append(dict(mode=mode,pooled_gain=gain,eligible_for_low_cost_learning=eligible,rows=rows))
summary=dict(training_only=True,teachers=results,optimizer_updates=0,goal_complete=False,acceptance_claim=False,expensive_training_allowed=False)
(OUT/'teacher_preflight_complete.json').write_text(json.dumps(summary,indent=2));print('TEACHER_PREFLIGHT',json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results]),flush=True)
