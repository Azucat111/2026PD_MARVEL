"""Detached same-cohort supervisor; never change guarded production sources."""
from pathlib import Path
import fcntl,time,json,subprocess,sys,os,hashlib
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19'
lock=(folder/'detached_supervisor.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(0)
def running(name):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:parts=(p/'cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(x.decode(errors='replace')==name for x in parts):return True
 return False
state=folder/'detached_supervisor_state.json'
def save(phase,**more):
 state.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),training_cohort_unchanged=True,**more),indent=2));print(phase,more,flush=True)
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261013_attempt19.json').read_text())
def check():
 assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items())
check();save('SUPERVISING_RUNNING_SAME_COHORT')
while not (folder/'complete.json').exists():
 if running('scripts/resume_opportunity_attempt19.py') or running('scripts/train_extended_gppo_public_opportunity_attempt19.py'):
  time.sleep(30);continue
 check();save('RESUMING_SAVED_MODEL_AND_OPTIMIZER')
 with (folder/f'supervised_resume_{int(time.time())}.log').open('x') as log:
  result=subprocess.run([sys.executable,'-u','-B','scripts/resume_opportunity_attempt19.py'],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:save('RESUME_FAILED',exit_code=result.returncode);sys.exit(result.returncode)
check()
if (folder/'development_9170_complete.json').exists():save('DEVELOPMENT_ALREADY_FINISHED');sys.exit(0)
save('AUDIT_AND_FIXED_FINAL_CANDIDATE_DEVELOPMENT')
result=subprocess.run([sys.executable,'-u','-B','scripts/continue_completed_opportunity_attempt19.py'])
save('PIPELINE_FINISHED' if result.returncode==0 else 'PIPELINE_FAILED',exit_code=result.returncode)
sys.exit(result.returncode)
