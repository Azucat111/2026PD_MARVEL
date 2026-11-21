from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_heat_facing_probe_20261015_attempt32'
plan=json.loads((OUT/'plan.json').read_text());index=json.loads((ROOT/'artifacts/extended_public_visibility_preflight_20261015_attempt24/selected_branches.json').read_text())
def guard():
 for n,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
def run(j):
 seed,c=j;p=OUT/'data'/f'seed_{seed}_heat'
 if not (p/'complete.json').exists():
  assert not p.exists(),f'Preserving incomplete {p}'
  cmd=[sys.executable,'-u','-B','scripts/evaluate_heat_facing_probe_branch.py','--seed',str(seed),'--uavs',str(c['uavs']),'--mode','heat']
  if c['dynamic']:cmd.append('--dynamic')
  with (OUT/f'branch_{seed}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(seed,r.returncode)
 r=json.loads((p/'complete.json').read_text());assert r['bounded_window_complete'] and not r['policy_failures']
 base=ROOT/index[f'{seed}:0']['folder']
 for f in ['config.json','initial.json']:assert json.loads((base/'rule'/f).read_text())==json.loads((p/'rule'/f).read_text()),(seed,f)
 b=json.loads((base/'complete.json').read_text())
 guard();row=dict(seed=seed,baseline=b['metric'],candidate=r['metric'],collisions=r['collision_types'],heading_changes=len(r['heading_changes']))
 print('HEADING_PAIRED',json.dumps(row),flush=True);return row
guard()
with ThreadPoolExecutor(max_workers=plan['workers']) as ex:rows=list(ex.map(run,zip(plan['training_seeds'],plan['case_cycle'])))
gain=sum(r['baseline']['mean_capped_seconds'] for r in rows)/sum(r['candidate']['mean_capped_seconds'] for r in rows)-1
r=dict(rows=rows,pooled_gain=gain,all_found=all(r['candidate']['detected_targets']==4 for r in rows),all_zero_collisions=all(not r['collisions'] for r in rows),goal_complete=False,acceptance_claim=False,optimizer_updates=0)
(OUT/'complete.json').write_text(json.dumps(r,indent=2));print('HEADING_COMPLETE',json.dumps(r),flush=True)
