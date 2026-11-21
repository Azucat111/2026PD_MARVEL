"""Resumable bounded physical paired-label collection; no optimizer updates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20'
plan=json.loads((OUT/'plan.json').read_text())
def guard():
 for name,digest in plan['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 for name,digest in plan['protected_weights'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
def run(job):
 seed,n,dynamic,rank=job;p=OUT/'data'/f'seed_{seed}_rank_{rank}'
 if (p/'complete.json').exists():return json.loads((p/'complete.json').read_text())
 assert not p.exists(),f'Incomplete branch retained; explicit recovery needed: {p}'
 cmd=[sys.executable,'-u','-B','scripts/collect_paired_calibration_branch.py','--seed',str(seed),'--uavs',str(n),'--rank',str(rank)]
 if dynamic:cmd.append('--dynamic')
 with (OUT/f'seed_{seed}_rank_{rank}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(job,r.returncode)
 result=json.loads((p/'complete.json').read_text())
 assert result['full_short_window'] and not result['policy_failures'],job
 print('COLLECTED',seed,rank,result['metric']['mean_capped_seconds'],flush=True)
 return result
jobs=[]
for i,seed in enumerate(plan['training_seeds']+plan['validation_seeds']):
 c=plan['case_cycle'][i%4]
 for rank in plan['candidate_ranks']:jobs.append((seed,c['uavs'],c['dynamic'],rank))
guard()
with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
 futures=[pool.submit(run,j) for j in jobs]
 for f in as_completed(futures):f.result();guard()
usable=[];censored=[]
for seed in plan['training_seeds']+plan['validation_seeds']:
 first=None;initial=None;configs=None;rows=[]
 for rank in plan['candidate_ranks']:
  p=OUT/'data'/f'seed_{seed}_rank_{rank}'
  f=json.loads((p/'first_decision.json').read_text());r=json.loads((p/'complete.json').read_text());init=json.loads((p/'rule/initial.json').read_text());cfg=json.loads((p/'rule/config.json').read_text())
  if first is None:first=f;initial=init;configs=cfg
  assert f['graph']==first['graph'] and f['actor_batch']==first['actor_batch'] and f['top_public_options']==first['top_public_options'] and init==initial and cfg==configs,seed
  assert f['warm_sha256']==first['warm_sha256'] and r['full_short_window'] and not r['policy_failures']
  rows.append(dict(rank=rank,mean_find=r['metric']['mean_capped_seconds'],found=r['metric']['detected_targets'],collisions=sum(r['collision_types'].values())))
 if any(r['found']<4 for r in rows):censored.append(dict(seed=seed,rows=rows))
 else:usable.append(dict(seed=seed,rows=rows))
result=dict(complete=True,training_updates=0,acceptance_claim=False,matched_states=True,source_stable=True,branches=len(jobs),usable_seeds=usable,censored_seeds=censored)
(OUT/'collection_complete.json').write_text(json.dumps(result,indent=2));print('COLLECTION_COMPLETE',json.dumps(result),flush=True)
