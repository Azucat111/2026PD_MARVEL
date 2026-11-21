"""Reserved-seed final only after fixed independent development gate passes."""
from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from score_gppo_pairs import load_case,aggregate_cases
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19';dev=json.loads((folder/'development_9170_complete.json').read_text())
assert dev['eligible_for_reserved_final'], 'Independent development gate did not pass; reserved data must remain untouched'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261013_attempt19.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['source_hashes'].items())
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==manifest['frozen_checkpoint_sha256']
assert hashlib.sha256((ROOT/'load_model/MARVEL/checkpoint.pth').read_bytes()).hexdigest()==manifest['native_checkpoint_sha256']==plan['native_checkpoint_sha256']
regression=json.loads((ROOT/'artifacts/extended_relay_known_route_design_20261013_attempt1/installed_actual_regressions.json').read_text());assert regression['passed']
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in regression['source_hashes'].items())
candidate=folder/'checkpoint_batch_031.pth';digest=hashlib.sha256(candidate.read_bytes()).hexdigest();assert all(r['checkpoint_sha256']==digest for r in dev['rows'])
output=ROOT/'artifacts/extended_integrated_reserved_final_20261013_attempt19';output.mkdir(exist_ok=False)
cases=[dict(seed=seed,uavs=case['uavs'],domain=case['domain']) for seed in plan['final_reserved_seeds'] for case in plan['development']['cases']]
with (output/'predeclared_plan.json').open('x') as s:json.dump(dict(cases=cases,horizon_seconds=300,checkpoint_sha256=digest,primary_threshold=.15,common_wall_seconds=3000,source_stable=True,strong_rule_unchanged=True,acceptance_claim=False),s,indent=2)
rows=[];joint=[]
for case in cases:
 seed=case['seed'];n=case['uavs'];domain=case['domain'];relative=f'artifacts/extended_integrated_final_{seed}_{n}_{domain}_20261013_attempt19';out=ROOT/relative
 args=[sys.executable,'-B','scripts/evaluate_opportunity_gppo_joint_acceptance.py','--checkpoint',str(candidate),'--out',relative,'--uavs',str(n),'--seed',str(seed),'--steps','300','--final','--scenario',plan['scenario'],'--wall-budget-seconds','3000','--warning-rate-mps','.2']
 if domain=='dynamic':args.append('--dynamic')
 print('RESERVED_FINAL_START',seed,n,domain,flush=True);subprocess.run(args,cwd=ROOT,check=True)
 assert json.loads((out/'manifest.json').read_text())['frozen_checkpoint_sha256']==manifest['frozen_checkpoint_sha256']
 rows.append(load_case(out));summary=json.loads((out/'gppo/summary.json').read_text());joint.append(dict(seed=seed,uavs=n,domain=domain,**summary['secondary_joint_metric']))
 result=dict(rows=rows,score=aggregate_cases(rows),joint=joint,project_goal_complete=False)
 with (output/f'partial_{len(rows):02d}.json').open('x') as s:json.dump(result,s,indent=2)
 print('RESERVED_FINAL_COMPLETE',json.dumps(rows[-1]),flush=True)
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['source_hashes'].items())
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==manifest['frozen_checkpoint_sha256']
result['project_goal_complete']=bool(result['score']['final_search_metric_acceptance'] and all(j['joint_success'] and j['whole_window_safe'] for j in joint))
result['limitations']='Predeclared connected 80m observed domain and three independent reserved seeds; no global map/planning guarantee. Native regression and protected weights audited separately.'
with (output/'complete.json').open('x') as s:json.dump(result,s,indent=2)
print(json.dumps(result),flush=True)
