"""Capture eight first-allocation prefixes only; reuse existing outcome files separately."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_public_visibility_preflight_20261015_attempt24'
plan=json.loads((OUT/'plan.json').read_text())
def run(item):
 i,seed=item;p=OUT/'data'/f'seed_{seed}'/'public_state.json'
 if p.exists():return
 case=plan['case_cycle'][i%4];cmd=[sys.executable,'-u','-B','scripts/capture_public_search_visibility_state.py','--seed',str(seed),'--uavs',str(case['uavs']),'--rank','0']
 if case['dynamic']:cmd.append('--dynamic')
 with (OUT/f'capture_{seed}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0 and p.exists(),seed
 print('PUBLIC_CAPTURED',seed,flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
 for f in as_completed([pool.submit(run,item) for item in enumerate(plan['training_seeds'])]):f.result()
OLD=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20';index=json.loads((OLD/'selected_branches.json').read_text());matches=[]
for seed in plan['training_seeds']:
 state=json.loads((OUT/'data'/f'seed_{seed}'/'public_state.json').read_text());old=json.loads((OLD/index[f'{seed}:0']/'first_decision.json').read_text())
 match=all(state[k]==old[k] for k in ['graph','actor_batch','top_public_options'])
 matches.append(dict(seed=seed,public_state_matches=match,known_cells=len(state['public_visibility']['observed_cells'])))
 assert match,seed
result=dict(complete=True,matches=matches,optimizer_updates=0,acceptance_claim=False)
(OUT/'public_capture_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
