"""Fixed paired policy validation on fresh seeds; no fitting or outcome selection."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_second_search_preflight_20261015_attempt21'
plan=json.loads((OUT/'plan.json').read_text());freeze=json.loads((OUT/'candidate_frozen_before_validation.json').read_text())
assert hashlib.sha256((OUT/'candidate.pth').read_bytes()).hexdigest()==freeze['candidate_sha256']
def folder(seed,mode):
 base=OUT/'validation'/f'seed_{seed}_{mode}'
 paths=[p for p in [base,base.with_name(base.name+'_retry_01')] if (p/'complete.json').exists()]
 assert len(paths)<=1
 return paths[0] if paths else base
def run(job):
 seed,n,dynamic,mode=job;p=folder(seed,mode)
 if (p/'complete.json').exists():return
 assert not p.with_name(p.name+'_retry_01').exists()
 cmd=[sys.executable,'-u','-B','scripts/validate_second_search_calibration_branch.py','--seed',str(seed),'--uavs',str(n),'--rank','0','--mode',mode]
 if dynamic:cmd.append('--dynamic')
 with (OUT/f'validation_{seed}_{mode}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(job,r.returncode)
 result=json.loads((folder(seed,mode)/'complete.json').read_text())
 assert result['bounded_window_complete'] and not result['policy_failures']
 print('VALIDATED',seed,mode,result['metric']['mean_capped_seconds'],flush=True)
jobs=[]
for i,seed in enumerate(plan['fresh_validation_seeds']):
 c=plan['case_cycle'][i%4]
 for mode in ['baseline','candidate']:jobs.append((seed,c['uavs'],c['dynamic'],mode))
with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):future.result()
rows=[]
for seed in plan['fresh_validation_seeds']:
 b=folder(seed,'baseline');c=folder(seed,'candidate')
 for name in ['config.json','initial.json']:assert json.loads((b/'rule'/name).read_text())==json.loads((c/'rule'/name).read_text())
 bs=json.loads((b/'search_decisions.json').read_text());cs=json.loads((c/'search_decisions.json').read_text())
 for key in ['graph','actor_batch','top_public_options']:assert bs[0][key]==cs[0][key],(seed,key)
 baseline=json.loads((b/'complete.json').read_text());candidate=json.loads((c/'complete.json').read_text())
 rows.append(dict(seed=seed,baseline=baseline['metric'],candidate=candidate['metric'],baseline_collisions=baseline['collision_types'],candidate_collisions=candidate['collision_types'],chosen_ranks=[r['rank'] for r in cs]))
gain=sum(r['baseline']['mean_capped_seconds'] for r in rows)/sum(r['candidate']['mean_capped_seconds'] for r in rows)-1
eligible=gain>=.15 and all(r['baseline']['detected_targets']==4 and r['candidate']['detected_targets']==4 and not r['baseline_collisions'] and not r['candidate_collisions'] for r in rows)
assert hashlib.sha256((OUT/'candidate.pth').read_bytes()).hexdigest()==freeze['candidate_sha256']
result=dict(candidate_sha256=freeze['candidate_sha256'],validation=rows,pooled_gain=gain,eligible_for_h300_development=eligible,acceptance_claim=False,goal_complete=False,short_sensor_find_validation_only=True)
(OUT/'validation_complete.json').write_text(json.dumps(result,indent=2));print('INDEPENDENT_VALIDATION',json.dumps(result),flush=True)
