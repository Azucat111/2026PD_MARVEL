"""Stop only the idle future experiment waiter, before any training exists."""
from pathlib import Path
import os,signal,json
ROOT=Path(__file__).resolve().parents[1]
assert not (ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt16').exists(), 'Do not interrupt training'
log=(ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt16_pipeline_stdout.log').read_text()
assert log.strip()=='WAITING_FOR_COMPLETE_ATTEMPT15_INDEPENDENT_DEVELOPMENT_GATE',log
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(bytes([0]))
 except (OSError,PermissionError):continue
 if b'scripts/wait_and_run_role_normalized_training_attempt16.py' in args and Path(args[0].decode()).name=='python':matches.append(int(p.name))
assert len(matches)==1,matches
os.kill(matches[0],signal.SIGINT)
result=dict(stopped_idle_waiter_pid=matches[0],training_started=False,reason='Preserve pending plan16; prepare equivalent sensing computation optimization before new training. Current attempt15 independent evaluation remains untouched.',all_files_preserved=True,production_changed=False,goal_complete=False)
(ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt16_idle_waiter_superseded.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
