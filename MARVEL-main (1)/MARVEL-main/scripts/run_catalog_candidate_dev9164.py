from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from score_gppo_pairs import load_case,aggregate_cases
folder=root/'artifacts/extended_gppo_joint_batch_20261010_attempt7'
assert json.loads((folder/'post_training_source_guard.json').read_text())['passed']
assert json.loads((folder/'rollout_audit_complete16.json').read_text())['passed']
plan=json.loads((root/'artifacts/extended_catalog_candidate_development_plan_9164_20261010.json').read_text());assert plan['seed']==9164 and plan['steps']==80
rows=[]
for case in plan['cases']:
 n=case['uavs'];domain=case['domain'];out=f'artifacts/extended_catalog_candidate_9164_{n}_{domain}_20261010_attempt1'
 args=[sys.executable,'-B','scripts/evaluate_extended_gppo_joint_acceptance.py','--checkpoint',str(folder/'checkpoint_batch_001.pth'),'--out',out,'--uavs',str(n),'--seed','9164','--steps','80','--scenario',plan['scenario'],'--wall-budget-seconds','600','--warning-rate-mps','0.2']
 if domain=='dynamic':args.append('--dynamic')
 print('PAIRED_DEVELOPMENT_START',n,domain,flush=True)
 subprocess.run(args,cwd=root,check=True)
 rows.append(load_case(root/out));print('PAIRED_DEVELOPMENT_COMPLETE',json.dumps(rows[-1]),flush=True)
 result=dict(cases=rows,score=aggregate_cases(rows),acceptance_claim=False)
 with (root/'artifacts/extended_catalog_candidate_9164_balanced4_score_20261010_attempt1.json').open('w') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result),flush=True)
