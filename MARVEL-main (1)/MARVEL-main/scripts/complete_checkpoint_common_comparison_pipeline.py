"""Finish the user-requested evaluation automatically after the frozen cohort."""
from pathlib import Path
import time,os,json,subprocess,sys,datetime
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
def status(phase,**extra):
 (OUT/'diagnosis_pipeline_status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_updates=0,**extra),indent=2))
status('WAITING_FROZEN_48_RUN_COMPARISON')
while not (OUT/'comparison_complete.json').exists():
 try:os.kill(751226,0)
 except ProcessLookupError:
  status('FAILED',reason='Frozen comparison parent exited before complete artifact');raise SystemExit(1)
 time.sleep(15)
for script,phase in [('finish_checkpoint_common_diagnosis.py','MATCHED_REPLAY_AND_CAUSAL_DIAGNOSIS'),('summarize_checkpoint_common_diagnosis.py','FINAL_EVIDENCE_VERIFICATION')]:
 status(phase)
 log=OUT/(script.removesuffix('.py')+'.log')
 with log.open('x') as f:r=subprocess.run([sys.executable,'-u','-B','scripts/'+script],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  status('FAILED',reason=script,exit_code=r.returncode,log=str(log));raise SystemExit(r.returncode)
status('COMPLETE',result='final_diagnosis.json',training_started=False)
p=ROOT/'artifacts/extended_training_execution_progress_20261010.json';progress=json.loads(p.read_text());progress['active_bounded_probe']=None;progress['checkpoint_common_comparison_complete']=True;progress['checkpoint_common_comparison_result']=str((OUT/'final_diagnosis.json').relative_to(ROOT));progress['training_permitted']=False;progress['project_goal_complete']=False;progress['status']='User-requested common checkpoint comparison and bounded causal diagnosis COMPLETE; no training performed';p.write_text(json.dumps(progress,indent=2))
print('USER_REQUESTED_COMPARISON_COMPLETE',flush=True)
