"""Bounded fresh paired diagnostic; no training or production mutation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/extended_counterfactual_public_probe_20261014';OUT.mkdir(exist_ok=False)
FILES=list((ROOT/'integrations').rglob('*.py'))+list((ROOT/'utils').rglob('*.py'))
sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in FILES}
plan=dict(diagnostic_only=True,acceptance_claim=False,seeds=[63380,63381],cases=[dict(seed=63380,uavs=30,dynamic=False),dict(seed=63381,uavs=60,dynamic=True)],public_choice_ranks=[0,1,2,3],horizon_seconds=30,workers=2,source_hashes=sources,reserved_seeds_touched=False,selection='priority, observed raw total travel/processing cost, stable public task and UAV IDs; change only first Search assignment')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2))
def run(job):
 seed,n,dynamic,rank=job
 cmd=[sys.executable,'-u','-B','scripts/probe_public_first_assignment.py','--seed',str(seed),'--uavs',str(n),'--rank',str(rank)]
 if dynamic:cmd.append('--dynamic')
 with (OUT/f'seed_{seed}_rank_{rank}.log').open('x') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(job,r.returncode)
 result=json.loads((OUT/f'seed_{seed}_rank_{rank}'/'complete.json').read_text())
 print('PAIRED_BRANCH',json.dumps(dict(seed=seed,rank=rank,metric=result['metric'])),flush=True)
 return result
jobs=[(63380,30,False,k) for k in range(4)]+[(63381,60,True,k) for k in range(4)]
with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in sources.items())
results=[]
for seed in plan['seeds']:
 baseline=json.loads((OUT/f'seed_{seed}_rank_0'/'complete.json').read_text())
 first=json.loads((OUT/f'seed_{seed}_rank_0'/'first_decision.json').read_text())
 initial=json.loads((OUT/f'seed_{seed}_rank_0'/'rule/initial.json').read_text())
 for k in range(4):
  p=OUT/f'seed_{seed}_rank_{k}';r=json.loads((p/'complete.json').read_text());f=json.loads((p/'first_decision.json').read_text())
  assert f['graph']==first['graph'] and f['top_public_options']==first['top_public_options']
  assert json.loads((p/'rule/initial.json').read_text())==initial
  gain=baseline['metric']['mean_capped_seconds']/r['metric']['mean_capped_seconds']-1
  results.append(dict(seed=seed,rank=k,choice=f['selected'],mean_find=r['metric']['mean_capped_seconds'],gain_vs_rule=gain,found=r['metric']['detected_targets'],collisions=sum(r['collision_types'].values()),full_short_window=r['full_short_window']))
(OUT/'complete.json').write_text(json.dumps(dict(diagnostic_only=True,acceptance_claim=False,source_stable=True,matched_initial_and_first_graph=True,rows=results,limitations='two seeds, one changed decision, H30 only; no H300 joint acceptance or learned deployment'),indent=2))
print('BOUNDED_PROBE_COMPLETE',json.dumps(results),flush=True)
