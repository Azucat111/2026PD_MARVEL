"""Predeclared fresh training, fixed final candidate, then common H300 development."""
from pathlib import Path
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from score_gppo_pairs import load_case,aggregate_cases
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt10.json').read_text())
installed=ROOT/'artifacts/extended_integrated_candidate_design_20261011_attempt1'
assert json.loads((installed/'actual_production_tests.json').read_text())['passed']
assert json.loads((installed/'safety_capture_actual_tests.json').read_text())['passed']
smoke=ROOT/'artifacts/extended_integrated_natural_61206_30_dynamic_20261011_attempt1'
assert json.loads((smoke/'comparison.json').read_text())['valid']
assert json.loads((smoke/'gppo/summary.json').read_text())['metric']['detected_targets']==4
initial=ROOT/plan['initial'];assert hashlib.sha256(initial.read_bytes()).hexdigest()==plan['initial_sha256']
args=[sys.executable,'-B','scripts/train_extended_gppo_wall_budget.py','--training-wall-budget-seconds',str(plan['training_wall_budget_seconds']),'--initial',str(initial),'--out',plan['out'],'--seeds',*map(str,plan['training_seeds']),'--steps',str(plan['horizon_seconds']),'--lr',str(plan['lr']),'--ppo-epochs',str(plan['ppo_epochs']),'--temperature',str(plan['temperature']),'--rollout-batch-size',str(plan['rollout_batch_size']),'--target-kl',str(plan['target_kl']),'--anchor-weight',str(plan['anchor_weight']),'--entropy-weight',str(plan['entropy_weight']),'--scenario',plan['scenario'],'--balanced-domains','--warning-rate-mps',str(plan['warning_rate_mps']),'--joint-rollout','--relay-reward-weight',str(plan['relay_reward_weight'])]
assert json.loads((ROOT/plan['continuation']['preceding_audit']).read_text())['passed']
assert len(json.loads((ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt9/updates.json').read_text()))==3
print('PREDECLARED_TRAINING_START',flush=True);subprocess.run(args,cwd=ROOT,check=True)
folder=ROOT/plan['out'];manifest=json.loads((folder/'manifest.json').read_text());complete=json.loads((folder/'complete.json').read_text());assert complete['episodes']==8 and complete['updates']==1 and complete['initial_unchanged']
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['source_hashes'].items())
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth');assert hashlib.sha256(frozen.read_bytes()).hexdigest()==manifest['frozen_checkpoint_sha256']
subprocess.run([sys.executable,'-B','scripts/audit_joint_ppo_rollouts.py','--folder',plan['out'],'--episodes','8','--out',plan['out']+'/rollout_audit_complete8.json'],cwd=ROOT,check=True)
candidate=folder/'checkpoint_batch_000.pth';candidate_hash=hashlib.sha256(candidate.read_bytes()).hexdigest()
with (folder/'post_training_source_guard.json').open('x') as s:json.dump(dict(passed=True,episodes=8,updates=1,preceding_used_episodes=24,preceding_updates=3,source_stable=True,frozen_unchanged=True,fixed_last_candidate_sha256=candidate_hash,acceptance_claim=False),s,indent=2)
rows=[];joint=[]
for case in plan['development']['cases']:
 n=case['uavs'];domain=case['domain'];relative=f'artifacts/extended_integrated_candidate_9165_{n}_{domain}_20261011_attempt1';out=ROOT/relative
 args=[sys.executable,'-B','scripts/evaluate_integrated_gppo_joint_acceptance.py','--checkpoint',str(candidate),'--out',relative,'--uavs',str(n),'--seed','9165','--steps','300','--scenario',plan['scenario'],'--wall-budget-seconds','3000','--warning-rate-mps','.2']
 if domain=='dynamic':args.append('--dynamic')
 print('H300_PAIRED_DEVELOPMENT_START',n,domain,flush=True);subprocess.run(args,cwd=ROOT,check=True)
 rows.append(load_case(out));summary=json.loads((out/'gppo/summary.json').read_text());joint.append(dict(uavs=n,domain=domain,**summary['secondary_joint_metric']))
 result=dict(rows=rows,score=aggregate_cases(rows),joint=joint,acceptance_claim=False)
 with (folder/f'development_partial_{len(rows):02d}.json').open('x') as s:json.dump(result,s,indent=2)
 print('H300_PAIRED_DEVELOPMENT_COMPLETE',json.dumps(rows[-1]),flush=True)
result['eligible_for_reserved_final']=all(r['valid'] and r['collisions']==0 and r['found']==r['targets'] for r in rows) and result['score']['efficiency_gain']>=.15 and all(j['joint_success'] for j in joint)
with (folder/'development_9165_complete.json').open('x') as s:json.dump(result,s,indent=2)
print(json.dumps(result),flush=True)

if result['eligible_for_reserved_final']:
 subprocess.run([sys.executable,'-B','scripts/run_integrated_reserved_final_attempt10_20261011.py'],cwd=ROOT,check=True)
else:
 with (folder/'reserved_final_deferred.json').open('x') as s:json.dump(dict(reason='Independent fixed candidate development gate rejected',final_reserved_touched=False,primary_gain=result['score']['efficiency_gain'],joint=result['joint']),s,indent=2)
 print('FINAL_RESERVED_UNTOUCHED_DEVELOPMENT_GATE_REJECTED',flush=True)
