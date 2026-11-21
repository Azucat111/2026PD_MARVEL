"""Install only validated observed Relay recovery, freeze sources, then fresh training."""
from pathlib import Path
import sys,json,time,hashlib,subprocess,ast
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt2';OLD=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
conditions=[OLD/'development_9168_complete.json',OLD/'reserved_final_deferred.json',STAGE/'staged_full_platform_regressions_v2.json',STAGE/'natural_30_static_62811_recovery150.json',STAGE/'natural_30_dynamic_62813_recovery150.json']
print('WAITING_VALIDATED_RECOVERY_AND_COMPLETED_ATTEMPT17',flush=True)
while not all(p.exists() for p in conditions):time.sleep(20)
dev=json.loads(conditions[0].read_text());assert not dev['eligible_for_reserved_final'];assert json.loads(conditions[1].read_text())['final_reserved_touched'] is False
reg=json.loads(conditions[2].read_text());assert reg['passed'] and reg['production_source_stable']
eq=json.loads((STAGE/'initialized_runtime_guard_equivalence.json').read_text());assert eq['passed'] and eq['new_scheduler_sha256']==digest(STAGE/'scheduler_draft.py') and eq['old_scheduler_sha256']==digest(STAGE/'scheduler_before_partial_fixture_guard.py')
for file in conditions[3:]:
 proof=json.loads(file.read_text());assert proof['secondary_joint_metric']['joint_success'] and proof['secondary_joint_metric']['whole_window_safe'] and not proof['summary']['collision_types'];assert proof['production_source_stable']
 for n,h in proof['staged_source_hashes'].items():
  assert (h==eq['old_scheduler_sha256'] if n=='scheduler_draft.py' else digest(STAGE/n)==h)
for n,h in reg['staged_hashes'].items():assert digest(STAGE/n)==h
oldplan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt17.json').read_text())
for p,h in {**oldplan['source_hashes'],**oldplan['pipeline_hashes']}.items():assert digest(ROOT/p)==h,p
assert not (ROOT/'artifacts/extended_integrated_reserved_final_20261012_attempt17').exists()
assert not (ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18').exists()
changes=[('scheduler_draft.py','integrations/gppo/scheduler.py'),('policy_adapter_draft.py','utils/policy_adapter.py'),('relay_frontier.py','integrations/gppo/relay_frontier.py'),('test_observed_relay_probe_installed.py','platform_tests/test_observed_relay_probe.py')]
backup=STAGE/'preinstallation_source_backup';backup.mkdir(exist_ok=False);installation=[]
for draft,relative in changes:
 target=ROOT/relative;old=digest(target) if target.exists() else None
 if target.exists():
  saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(target.read_bytes())
 target.write_bytes((STAGE/draft).read_bytes());installation.append(dict(file=relative,before=old,after=digest(target)))
(STAGE/'installation_attempt18.json').write_text(json.dumps(dict(changes=installation,old_gate_primary=dev['score']['efficiency_gain'],native_unchanged=True,reserved_final_touched=False,acceptance_claim=False),indent=2))
print('VALIDATED_RECOVERY_INSTALLED',flush=True)
subprocess.run([sys.executable,'-u','-B','scripts/verify_installed_relay_probe_regressions.py'],cwd=ROOT,check=True)
installed=json.loads((STAGE/'installed_actual_regressions.json').read_text());assert installed['passed']
assert installed['native_checkpoint_sha256']==oldplan['native_checkpoint_sha256']
assert installed['frozen_gppo_sha256']=='57abf685f7a8226dcf71e1441333bf74da6e228e423f1bb0034e110975e3fcd9'
keys=['scenario','warning_rate_mps','lr','ppo_epochs','temperature','rollout_batch_size','target_kl','anchor_weight','entropy_weight','joint_rollout','relay_reward_weight','advantage_normalization','native_checkpoint_sha256','training_clock_tests','worker_limit','training_wall_budget_seconds','physical_validation']
plan={k:oldplan[k] for k in keys};plan.update(declared_before_training=True,stage='fresh256_role_normalized_H150_after_observed_relay_recovery',initial='artifacts/extended_gppo_joint_batch_20261012_attempt17/checkpoint_batch_015.pth',initial_sha256='1cda34ecb47ec1eb411e3bf47f30351bbec1a7d48b96ff6297f70ac0739cf22a',out='artifacts/extended_gppo_joint_batch_20261012_attempt18',horizon_seconds=150,training_seeds=list(range(62820,63076)),balanced_domain_cycle=['30 static','60 static','30 dynamic','60 dynamic'],reset_critic=True,reset_scope='critic final value output and fresh Adam optimizer; trained actor/shared encoder preserved',fixed_candidate='predeclared last checkpoint_batch_031.pth only',final_reserved_seeds=[101101,202202,303303],final_horizon_seconds=300,acceptance_claim=False,regressions='artifacts/extended_relay_frontier_design_20261012_attempt2/installed_actual_regressions.json',continuation=dict(preceding_audit='artifacts/extended_gppo_joint_batch_20261012_attempt17/rollout_audit_complete128.json',reason='All256onpolicy episodes fresh; no prior transitions reused;H80->H150 requires explicit critic-output/optimizer reset'),evaluation_data_excluded_from_training=[9165,9166,9167,9168,9169,62811,62813,101101,202202,303303],relay_recovery='Observed-only post-required-task bounded information gathering; current helpers retained until existing release gate proves release; real R.2,Shield,radio remain authoritative')
plan['development']={**oldplan['development'],'seed':9169};plan['source_hashes']={p:digest(ROOT/p) for p in oldplan['source_hashes']};plan['source_hashes'].update(installed['source_hashes']);plan['source_hashes']['platform_tests/test_observed_relay_probe.py']=digest(ROOT/'platform_tests/test_observed_relay_probe.py')
plan['pipeline_hashes']={p:digest(ROOT/p) for p in oldplan['pipeline_hashes']}
newpipeline=['scripts/train_extended_gppo_parallel_role_normalized_attempt18.py','scripts/run_integrated_training_dev9169_attempt18.py','scripts/run_integrated_reserved_final_attempt18_20261012.py','scripts/wait_install_relay_and_train_attempt18.py','scripts/verify_installed_relay_probe_regressions.py']
for p in newpipeline:plan['pipeline_hashes'][p]=digest(ROOT/p);ast.parse((ROOT/p).read_text(),filename=p)
assert digest(ROOT/plan['initial'])==plan['initial_sha256']
assert len(plan['training_seeds'])==256 and not set(plan['training_seeds']) & set(plan['evaluation_data_excluded_from_training'])
with (ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt18.json').open('x') as handle:json.dump(plan,handle,indent=2)
print('PREDECLARED_FRESH256_H150_TRAINING_READY',flush=True)
subprocess.run([sys.executable,'-u','-B','scripts/run_integrated_training_dev9169_attempt18.py'],cwd=ROOT,check=True)
