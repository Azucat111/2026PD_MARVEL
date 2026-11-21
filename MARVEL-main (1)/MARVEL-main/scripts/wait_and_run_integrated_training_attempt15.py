"""Continue authorized pipeline automatically once fresh execution pilots pass."""
from pathlib import Path
import sys,time,json,subprocess
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/extended_parallel_rollout_execution_20261011_attempt2'
print('WAITING_FOR_CURRENT_SOURCE_FROZEN_ROLLOUT_PILOTS',flush=True)
started=time.monotonic()
while not (folder/'pilot_execution_results.json').exists():
 if (folder/'pilot_driver_stderr.log').exists() and 'Traceback' in (folder/'pilot_driver_stderr.log').read_text():raise RuntimeError('Pilot driver failed; all files preserved, no training update')
 if time.monotonic()-started>7200:raise RuntimeError('Pilot readiness timeout; no inference or training acceptance claim')
 time.sleep(2)
proof=json.loads((folder/'pilot_execution_results.json').read_text());assert proof['passed']
for name in ['pilot_30_static','pilot_60_dynamic']:
 assert json.loads((folder/(name+'_math_audit.json')).read_text())['passed']
 assert json.loads((folder/name/'rollout_only_execution_proof.json').read_text())['passed']
print('CURRENT_SOURCE_EXECUTION_PILOTS_PASS_START_FIXED_CONTINUATION',flush=True)
subprocess.run([sys.executable,'-u','-B','scripts/run_integrated_training_dev9167_attempt15.py'],cwd=ROOT,check=True)
