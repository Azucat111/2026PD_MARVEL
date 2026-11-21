from pathlib import Path
import json,hashlib,datetime,ast
R=Path.cwd();O=R/'artifacts/gppo_learnability_two_hours_20261111';OLD=R/'artifacts/gppo_three_hour_20261110';PREV=R/'artifacts/gppo_learnability_hour_20261110'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'plan.json').exists()
P=json.loads((OLD/'training/batch_00/plan.json').read_text())
P.update(status='COLLECTION_READY',start_utc='2026-11-11T04:49:18+00:00',deadline_utc='2026-11-11T06:49:18+00:00',collection_deadline_utc='2026-11-11T05:32:00+00:00',runner_deadline_utc='2026-11-11T05:33:00+00:00',update_deadline_utc='2026-11-11T05:40:00+00:00',workers=4,wall_budget_seconds_per_arm=1500,behavior_checkpoint='artifacts/gppo_three_hour_20261110/training/batch_00/behavior.pth',tasks=[dict(index=i,global_index=2 if i<4 else 3,seed=1280022 if i<4 else 1880024,sample_seed=3981100+i,folder=str((O/f'worker_{i:02d}').relative_to(R)),origin='new_collection') for i in range(8)],scope='Training-case learnability diagnostic only: four fresh policy draws each30dynamic/60dynamic, plus four same-behavior old30static references; no acceptance/generalization claim',new_policy_updates_max=1,batch_size=12)
P['reference_tasks']=[dict(index=100+i,global_index=0,seed=1880021,sample_seed=2980100+i,folder=str((PREV/f'worker_{i:02d}').relative_to(R)),origin='read_only_reference') for i in range(4)]
P['update_tasks']=P['reference_tasks']+P['tasks']
P['update'].update(advantage='within_seed_leave_one_out_episode_completion_fixed_scale_0.05',critic_loss=0,temperature=1.,lr=2e-5,ppo_epochs=4,target_kl=.005)
P['source_hashes'].update({f:v['sha256'] for f,v in json.loads((OLD/'execution_assets.json').read_text())['files'].items()})
assert all(sha(R/f)==h for f,h in P['source_hashes'].items())
assert sha(R/P['behavior_checkpoint'])==P['behavior_sha256']
references={}
for T in P['reference_tasks']:
 D=R/T['folder'];proof=json.loads((D/'collection_proof.json').read_text());assert proof['seed']==T['seed'] and proof['global_index']==0 and proof['sample_seed']==T['sample_seed'] and proof['optimizer_updates']==0 and proof['source_unchanged']
 for rel in ['collection_proof.json','captured_transitions.pth','manifest.json','complete.json','progress.json','episode_000/physical_joint_trace.json','episode_000/gppo/initial.json']:
  f=D/rel;references[str(f.relative_to(R))]=sha(f)
P['read_only_reference_hashes']=references
collect=(PREV/'collect.py').read_text()
assert collect.count('bench.run=bind_wall_budget(bench.run,900.)')==1
collect=collect.replace('bench.run=bind_wall_budget(bench.run,900.)','bench.run=bind_wall_budget(bench.run,1500.)')
collect=collect.replace("T=P['tasks'][a.index];i=T['global_index'];deadline=", "T=next(t for t in P['tasks'] if t['index']==a.index);i=T['global_index'];deadline=")
collect=collect.replace('check_deadline()\nfor f,h', "check_deadline()\nassert hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()==(O/'plan_sha256.txt').read_text()\nfor f,h in P['controller_hashes'].items():assert hashlib.sha256((O/f).read_bytes()).hexdigest()==h,f\nfor f,h")
collect=collect.replace("ns['main']();row,ts=", "ns['main']();check_deadline();row,ts=")
collect=collect.replace("result=dict(**T,alignment=proof,", "assert row['seed']==T['seed'] and row['uavs']==(30 if i%2==0 else 60)\nassert row['domain']==('dynamic' if i%4>=2 else 'static')\nassert hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()==(O/'plan_sha256.txt').read_text()\nassert all(hashlib.sha256((O/f).read_bytes()).hexdigest()==h for f,h in P['controller_hashes'].items())\nresult=dict(**T,behavior_sha256=P['behavior_sha256'],behavior_state_fingerprint=P['behavior_state_fingerprint'],controller_hashes=P['controller_hashes'],alignment=proof,")
(O/'collect.py').write_text(collect)
runner='''from pathlib import Path
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
'''
(O/'run_collection.py').write_text(runner)
for name in ['group_baseline.py','test_group_baseline.py']:(O/name).write_bytes((PREV/name).read_bytes())
P['controller_hashes']={f:sha(O/f) for f in ['collect.py','run_collection.py','group_baseline.py']}
P['prepared_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(O/'plan.json').write_text(json.dumps(P,indent=2));(O/'plan_sha256.txt').write_text(sha(O/'plan.json'))
for f in O.glob('*.py'):ast.parse(f.read_text())
print(json.dumps(dict(status='READY',new_tasks=P['tasks'],read_only_reference_episodes=4,source_hashes=len(P['source_hashes']),controller_hashes=P['controller_hashes'],wall_budget=1500,collection_cutoff=P['collection_deadline_utc'],runner_cutoff=P['runner_deadline_utc'])))
