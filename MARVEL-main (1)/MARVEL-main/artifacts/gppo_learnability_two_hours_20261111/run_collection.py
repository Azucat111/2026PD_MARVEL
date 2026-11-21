from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,sys,time,json,datetime,os,signal,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];P=json.loads((O/'plan.json').read_text());deadline=datetime.datetime.fromisoformat(P['runner_deadline_utc']).timestamp();cutoff=datetime.datetime.fromisoformat(P['collection_deadline_utc']).timestamp()
assert hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()==(O/'plan_sha256.txt').read_text()
assert all(hashlib.sha256((O/f).read_bytes()).hexdigest()==h for f,h in P['controller_hashes'].items())
(O/'runner_identity.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time(),deadline=deadline)))
def run(t):
 i=t['index']
 if time.time()>=cutoff:return dict(index=i,status='NOT_STARTED_CUTOFF')
 log=(O/f'worker_{i:02d}.log').open('x');cmd=[sys.executable,'-u','-B',str(O/'collect.py'),'--index',str(i)];p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();start=time.time();stopped=False
 (O/f'process_{i:02d}.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,start=start)))
 while p.poll() is None:
  if time.time()>=deadline or time.time()-start>=P['wall_budget_seconds_per_arm']:
   stopped=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(1)
 r=dict(index=i,exit_code=p.returncode,wall_seconds=time.time()-start,terminated_by_budget=stopped);(O/f'execution_{i:02d}.json').write_text(json.dumps(r));print('REPEAT_FINISHED',r,flush=True);return r
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,P['tasks']))
(O/'execution.json').write_text(json.dumps(rows,indent=2))
