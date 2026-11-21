"""Install exact cache only after old evaluation quiesces, then verify before new learning."""
from pathlib import Path
import sys,json,time,hashlib,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'artifacts/extended_exact_sensor_cache_design_20261012_attempt1'
oldplan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt15.json').read_text())
def proof_ready():
 files=['full_platform_staged_regressions.json','actual_physics_equivalence.json','controlled_still_runtime_equality_and_cpu.json']
 return all((STAGE/f).exists() for f in files)
started=time.monotonic();print('WAIT_FOR_COMPLETE15_GATE_AND_STAGED_EXACT_CACHE_PROOFS',flush=True)
while not proof_ready() or not (ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15/reserved_final_deferred.json').exists():
 gate=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15/development_9167_complete.json'
 if gate.exists() and json.loads(gate.read_text())['eligible_for_reserved_final']:
  print('ATTEMPT15_PASSED_NO_SOURCE_CHANGE_OR_NEW_TRAINING',flush=True);sys.exit(0)
 if time.monotonic()-started>21600:raise RuntimeError('Readiness exceeded6h; no source install or new training')
 time.sleep(10)
for f in ['full_platform_staged_regressions.json','actual_physics_equivalence.json','controlled_still_runtime_equality_and_cpu.json']:assert json.loads((STAGE/f).read_text())['passed'],f
assert not (ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt16').exists()
for f,h in {**oldplan['source_hashes'],**oldplan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt17.json').read_text())
assert hashlib.sha256((ROOT/plan['initial']).read_bytes()).hexdigest()==plan['initial_sha256']
target=ROOT/'utils/sensor_models.py';draft=STAGE/'sensor_models_draft.py'
assert target.read_bytes()==(STAGE/'sensor_models_before.py').read_bytes()
assert hashlib.sha256(draft.read_bytes()).hexdigest()==plan['source_hashes']['utils/sensor_models.py']
before=hashlib.sha256(target.read_bytes()).hexdigest();temporary=target.with_name('.sensor_models_exact_cache_install17.tmp');assert not temporary.exists();temporary.write_bytes(draft.read_bytes());os.replace(temporary,target)
with (STAGE/'installation_attempt17.json').open('x') as handle:json.dump(dict(before_sha256=before,after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),backup=str((STAGE/'sensor_models_before.py').relative_to(ROOT)),default_disabled=True,native_bypassed=True,production_installed=True,acceptance_claim=False),handle,indent=2)
print('EXACT_CACHE_INSTALLED_AFTER_COMPLETE_PRIOR_GATE_BEGIN_ACTUAL_REGRESSIONS',flush=True)
subprocess.run([sys.executable,'-B','scripts/verify_installed_exact_sensor_cache_regressions.py'],cwd=ROOT,check=True)
assert json.loads((STAGE/'installed_actual_regressions_attempt1.json').read_text())['passed']
for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
assert hashlib.sha256((ROOT/'load_model/MARVEL/checkpoint.pth').read_bytes()).hexdigest()==plan['native_checkpoint_sha256']
print('ACTUAL_CACHE_REGRESSIONS_PASS_START_FRESH_NORMALIZED_TRAINING17',flush=True)
subprocess.run([sys.executable,'-u','-B','scripts/run_integrated_training_dev9168_attempt17.py'],cwd=ROOT,check=True)
