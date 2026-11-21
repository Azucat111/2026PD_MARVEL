"""Run only the fixed training-case diagnostic; never train or select models."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,subprocess,os,signal,time,datetime,json,hashlib,threading
from summarize_evaluation import summarize
D=Path(__file__).resolve().parent;O=D.parent;R=D.parents[2]

def main():
 P=json.loads((D/'plan.json').read_text())
 if hashlib.sha256((D/'plan.json').read_bytes()).hexdigest()!=(D/'protocol_sha256.txt').read_text().strip():raise ValueError('Frozen diagnostic plan hash mismatch')
 assert P['diagnostic_training_cases'] is False and not P['independent_acceptance'] and not P['model_selection_permitted']
 assert P['arms']==['rule','parent','candidate'] and len(P['jobs'])==9
 deadline=datetime.datetime.fromisoformat(P['parent_deadline_utc']).timestamp()
 runtime_deadline=datetime.datetime.fromisoformat(P['runtime_deadline_utc']).timestamp()
 lock=threading.Lock()
 assert hashlib.sha256((R/P['update_proof_path']).read_bytes()).hexdigest()==P['update_proof_sha256']
 def run(job):
  i,arm=job['case'],job['arm']
  if time.time()+180>=runtime_deadline:
   result=dict(case=i,arm=arm,status='NOT_STARTED_BUDGET',diagnostic_training_cases=P['diagnostic_training_cases'],independent_acceptance=False)
   (D/f'case_{i}_{arm}_execution.json').write_text(json.dumps(result,indent=2))
   with lock:
    report=summarize();print('DIAGNOSTIC_ARM_NOT_STARTED',json.dumps(result),flush=True)
   return result
  with (D/f'case_{i}_{arm}.log').open('x') as log:
   cmd=[sys.executable,'-u','-B',str(D/'evaluate_arm.py'),'--case',str(i),'--arm',arm,'--mode','original']
   worker=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.time();stopped=False
   (D/f'case_{i}_{arm}_process.json').write_text(json.dumps(dict(pid=worker.pid,command=cmd,started=start)))
   while worker.poll() is None:
    if time.time()>=deadline or time.time()-start>P['wall_budget_seconds_per_arm']+10:
     stopped=True;os.killpg(worker.pid,signal.SIGTERM)
     try:worker.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
     break
    time.sleep(1)
  result=dict(case=i,arm=arm,exit_code=worker.returncode,wall_seconds=time.time()-start,terminated_by_budget=stopped,diagnostic_training_cases=False,independent_acceptance=False)
  (D/f'case_{i}_{arm}_execution.json').write_text(json.dumps(result,indent=2))
  with lock:
   report=summarize()
   print('DIAGNOSTIC_ARM_FINISHED',json.dumps(dict(execution=result,completed=report['completed_arms'],expected=report['expected_arms'],latest=[r for r in report['rows'] if r['case']==i and r['arm']==arm])),flush=True)
  return result
 (D/'runner_identity.json').write_text(json.dumps(dict(pid=os.getpid(),deadline=deadline,worker_limit=4,diagnostic_training_cases=False)))
 with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,P['jobs']))
 (D/'execution.json').write_text(json.dumps(results,indent=2))
 summarize()
if __name__=='__main__':main()
