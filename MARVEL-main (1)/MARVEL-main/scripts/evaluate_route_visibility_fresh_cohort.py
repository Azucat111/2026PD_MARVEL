"""Fixed attempt26 teacher on four unused seeds, paired physical rule control."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/extended_route_visibility_fresh_probe_20261015_attempt31'
plan=json.loads((OUT/'plan.json').read_text())
def guard():
 for n,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
def run(job):
 seed,n,dynamic,mode=job;p=OUT/'data'/f'seed_{seed}_{mode}'
 if not (p/'complete.json').exists():
  assert not p.exists(),f'Incomplete branch preserved: {p}'
  cmd=[sys.executable,'-u','-B','scripts/evaluate_route_visibility_fresh_branch.py','--seed',str(seed),'--uavs',str(n),'--mode',mode]
  if dynamic:cmd.append('--dynamic')
  with (OUT/f'branch_{seed}_{mode}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(job,r.returncode)
 r=json.loads((p/'complete.json').read_text());assert r['bounded_window_complete'] and not r['policy_failures'],job
 guard();print('BRANCH',seed,mode,r['metric']['mean_capped_seconds'],r['collision_types'],flush=True)
jobs=[]
for i,seed in enumerate(plan['training_seeds']):
 c=plan['case_cycle'][i%4]
 for mode in ['rule','greedy']:jobs.append((seed,c['uavs'],c['dynamic'],mode))
guard()
with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
 for f in as_completed([pool.submit(run,j) for j in jobs]):f.result()
rows=[]
for seed in plan['training_seeds']:
 b=OUT/'data'/f'seed_{seed}_rule';c=OUT/'data'/f'seed_{seed}_greedy'
 for f in ['config.json','initial.json']:assert json.loads((b/'rule'/f).read_text())==json.loads((c/'rule'/f).read_text()),(seed,f)
 br=json.loads((b/'complete.json').read_text());cr=json.loads((c/'complete.json').read_text())
 rows.append(dict(seed=seed,baseline=br['metric'],candidate=cr['metric'],baseline_collisions=br['collision_types'],candidate_collisions=cr['collision_types']))
gain=sum(r['baseline']['mean_capped_seconds'] for r in rows)/sum(r['candidate']['mean_capped_seconds'] for r in rows)-1
r=dict(rows=rows,pooled_gain=gain,all_found=all(v['detected_targets']==4 for row in rows for v in [row['baseline'],row['candidate']]),all_zero_collisions=all(not row[k] for row in rows for k in ['baseline_collisions','candidate_collisions']),goal_complete=False,acceptance_claim=False,horizon=30,expensive_training_started=False)
(OUT/'complete.json').write_text(json.dumps(r,indent=2));print('FRESH_COMPLETE',json.dumps(r),flush=True)
