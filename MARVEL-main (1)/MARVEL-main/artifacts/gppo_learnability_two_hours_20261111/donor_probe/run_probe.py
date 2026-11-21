from pathlib import Path
import subprocess,sys,os,signal,time,json,datetime
D=Path(__file__).resolve().parent;R=D.parents[2];deadline=datetime.datetime.fromisoformat('2026-11-11T06:41:00+00:00').timestamp()
cmd=[sys.executable,'-u','-B',str(D/'evaluate_arm.py'),'--case','0','--arm','candidate','--mode','original']
with (D/'candidate.log').open('x') as log:
 p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.time();stopped=False
 (D/'process.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,start=start,deadline=deadline)))
 while p.poll() is None:
  if time.time()>=deadline or time.time()-start>910:
   stopped=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(1)
 (D/'execution.json').write_text(json.dumps(dict(exit_code=p.returncode,wall_seconds=time.time()-start,terminated_by_budget=stopped,development_causal_probe=True,independent_acceptance=False),indent=2))
