"""Run sealed cheap preflight/collection/one fit, never long training without gate."""
from pathlib import Path
import sys,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20'
def state(phase,**kw):
 (OUT/'pipeline_state.json').write_text(json.dumps(dict(phase=phase,goal_complete=False,**kw),indent=2))
state('COLLECTING_PAIRED_H20_LABELS')
with (OUT/'collection_stdout.log').open('a') as log:
 r=subprocess.run([sys.executable,'-u','-B','scripts/collect_paired_calibration_cohort.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:state('COLLECTION_FAILED',exit_code=r.returncode);raise SystemExit(r.returncode)
state('ONE_REGISTERED_LOW_COST_FIT')
with (OUT/'fit_stdout.log').open('a') as log:
 r=subprocess.run([sys.executable,'-u','-B','scripts/fit_paired_calibration_once.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:state('FIT_FAILED',exit_code=r.returncode);raise SystemExit(r.returncode)
result=json.loads((OUT/'validation_complete.json').read_text())
state('SHORT_VALIDATION_PASSED' if result['eligible_for_h300_development'] else 'SHORT_VALIDATION_REJECTED',gain=result['pooled_gain'],candidate_sha256=result['candidate_sha256'],long_training_started=False,production_changed=False)
