"""Preserve the running immutable cohort; recover only its CLI audit invocation."""
from pathlib import Path
import time,os,json,sys,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'artifacts/extended_integrated_training_plan_20261013_attempt19.json'
plan=json.loads(PLAN.read_text());folder=ROOT/plan['out'];parent_pid=375563
state_file=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19_continuation_state.json'
def save(phase,**more):
 state_file.write_text(json.dumps(dict(phase=phase,training_rerun=False,final_reserved_touched=False,project_goal_complete=False,**more),indent=2));print(phase,flush=True)
def parent_running():
 try:return b'run_integrated_training_dev9170_attempt19.py' in Path(f'/proc/{parent_pid}/cmdline').read_bytes()
 except FileNotFoundError:return False
save('WAITING_ORIGINAL_FIXED_COHORT',original_parent_pid=parent_pid)
while parent_running():time.sleep(30)
if not (folder/'complete.json').exists():save('ORIGINAL_PIPELINE_EXITED_BEFORE_COHORT_COMPLETE');raise RuntimeError('Training incomplete; no automatic recollection permitted')
complete=json.loads((folder/'complete.json').read_text());assert complete['episodes']==256 and complete['updates']==32
assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items())
if (folder/'development_9170_complete.json').exists():
 save('ORIGINAL_PIPELINE_ALREADY_COMPLETED_DEVELOPMENT');sys.exit(0)
save('CONTINUING_SAME_FIXED_COHORT_AUDITS_AND_DEVELOPMENT',checkpoint_sha256=hashlib.sha256((folder/'checkpoint_batch_031.pth').read_bytes()).hexdigest())
result=subprocess.run([sys.executable,'-u','-B','scripts/continue_completed_opportunity_attempt19.py'],cwd=ROOT)
save('CONTINUATION_FINISHED' if result.returncode==0 else 'CONTINUATION_FAILED',exit_code=result.returncode)
sys.exit(result.returncode)
