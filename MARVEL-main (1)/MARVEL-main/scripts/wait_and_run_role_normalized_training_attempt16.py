"""Autonomous next bounded experiment only after complete independent gate rejection."""
from pathlib import Path
import sys,time,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
previous=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15/development_9167_complete.json'
started=time.monotonic();print('WAITING_FOR_COMPLETE_ATTEMPT15_INDEPENDENT_DEVELOPMENT_GATE',flush=True)
while not previous.exists():
 if time.monotonic()-started>21600:raise RuntimeError('Prior gate not completed within6h; files retained')
 stderr=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15_pipeline_stderr.log'
 if stderr.exists() and 'Traceback' in stderr.read_text():raise RuntimeError('Prior pipeline failed; requires diagnosis before new training')
 time.sleep(10)
gate=json.loads(previous.read_text())
if gate['eligible_for_reserved_final']:
 print('ATTEMPT15_PASSED_CONTINUES_ITS_RESERVED_FINAL_NO_NEW_TRAINING',flush=True);sys.exit(0)
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt16.json').read_text())
for f,d in {**plan['source_hashes'],**plan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==d,f
assert not (ROOT/'artifacts/extended_integrated_reserved_final_20261011_attempt1').exists()
assert not (ROOT/'artifacts/extended_integrated_reserved_final_20261012_attempt1').exists()
print('COMPLETE_GATE_REJECTED_START_PREDECLARED_FRESH_ROLE_NORMALIZED_TRAINING',gate['score']['efficiency_gain'],flush=True)
subprocess.run([sys.executable,'-u','-B','scripts/run_integrated_training_dev9168_attempt16.py'],cwd=ROOT,check=True)
