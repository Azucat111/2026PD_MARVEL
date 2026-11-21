"""One-shot continuation of this local scientific pipeline; no repeated experiments."""
from pathlib import Path
import json,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt9';done=folder/'development_9165_complete.json';started=time.monotonic()
while not done.exists():
 if time.monotonic()-started>43200:raise TimeoutError('No final data collected: development did not finish within12h')
 alive=subprocess.run(['pgrep','-f','run_integrated_training_dev9165_attempt9.py'],capture_output=True,text=True).returncode==0
 if not alive:
  with (folder/'reserved_final_deferred.json').open('x') as s:json.dump(dict(reason='Training/development driver exited without complete independent gate',final_reserved_touched=False),s)
  raise RuntimeError('Driver exited before development gate completion')
 time.sleep(30)
dev=json.loads(done.read_text())
if not dev['eligible_for_reserved_final']:
 with (folder/'reserved_final_deferred.json').open('x') as s:json.dump(dict(reason='Independent fixed candidate development gate rejected',final_reserved_touched=False,primary_gain=dev['score']['efficiency_gain'],joint=dev['joint']),s,indent=2)
 print('FINAL_RESERVED_UNTOUCHED_DEVELOPMENT_GATE_REJECTED',flush=True);sys.exit(0)
print('DEVELOPMENT_GATE_PASS_BEGIN_PREDECLARED_RESERVED_FINAL',flush=True)
subprocess.run([sys.executable,'-B','scripts/run_integrated_reserved_final_20261011.py'],cwd=ROOT,check=True)
