from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_route_visibility_fresh_probe_20261015_attempt31'
plan=json.loads((OUT/'plan.json').read_text())
def run(j):
 seed,c=j;cmd=[sys.executable,'-u','-B','scripts/capture_route_visibility_fresh_prefix.py','--seed',str(seed),'--uavs',str(c['uavs'])]
 if c['dynamic']:cmd.append('--dynamic')
 with (OUT/f'prefix_{seed}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(seed,r.returncode)
 d=json.loads((OUT/'prefix_diagnostics'/str(seed)/'complete.json').read_text());row=d['decision'];best=d['scores'][0]
 print(json.dumps(dict(seed=seed,step=d['step'],decision=row,best=best,legal_scored=len(d['scores']))),flush=True)
with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,zip(plan['training_seeds'],plan['case_cycle'])))
