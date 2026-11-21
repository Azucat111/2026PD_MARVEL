"""Frozen round8 independent paired complete-mission evaluation; no training."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib,datetime,copy,math,itertools
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(OUT/'plan.json');protocol_hash=(OUT/'protocol_sha256.txt').read_text().strip()
def guard():
 assert sha(OUT/'plan.json')==protocol_hash
 for f,h in plan['source_hashes'].items():assert sha(ROOT/f)==h,f
 for m in plan['models'].values():assert sha(ROOT/m['path'])==m['sha256']
 assert sha(Path(plan['frozen_checkpoint']))==plan['frozen_sha256']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['head']
def path(i,arm):
 c=plan['cases'][i]
 return OUT/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/arm
def config_normalized(p):
 c=copy.deepcopy(read(p))
 for k in ['extended_checkpoint','extended_checkpoint_sha256']:c['task_scheduler'].pop(k,None)
 return c
def validate_case(i):
 folders=[path(i,a) for a in plan['arms']]
 if not all((p/'complete.json').exists() for p in folders):return None
 records=[read(p/'complete.json') for p in folders]
 real=[p/('rule' if r['arm']=='rule' else 'gppo') for p,r in zip(folders,records)]
 for filename,loader in [('initial.json',read),('config.json',config_normalized),('private_target_identity_diagnostic_only.json',read)]:
  values=[loader(p/filename) for p in real];assert all(x==values[0] for x in values),('Unequal paired state',i,filename)
 actors=[read(p/'actor_inputs_9.json') for p in real]
 assert all(actors),'No first actor records'
 def state(r):return {k:r[k] for k in ['time','uav_ids','task_ids','batch','raw_edge_seconds','public_context','raw_uav_features','raw_task_features','step']}
 assert all(state(x[0])==state(actors[0][0]) for x in actors),'Unequal first public decision state'
 for r in records:
  assert r['source_stable'] and r['optimizer_updates']==0 and r['summary']['full_window_complete']
 return dict(case=plan['cases'][i],paired_identity_verified=True,arms={r['arm']:dict(metric=r['summary']['metric'],joint=r['summary']['secondary_joint_metric'],collisions=r['summary']['collision_types'],nonrelay_complete=r['summary']['required_nonrelay_tasks_complete'],valid=r['summary']['valid_for_comparison'],wall_seconds=r['summary']['seconds']) for r in records})
def run(i,arm):
 p=path(i,arm)
 if not (p/'complete.json').exists():
  assert not p.exists(),f'Partial run preserved: {p}'
  log=OUT/'logs'/f'case_{i:02d}_{arm}.log';log.parent.mkdir(exist_ok=True)
  with log.open('x') as f:r=subprocess.run([sys.executable,'-u','-B',str(OUT/'evaluate_arm.py'),'--case',str(i),'--arm',arm],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,(i,arm,r.returncode,str(log))
 guard();r=read(p/'complete.json')
 print('ARM_VERIFIED',i,arm,'mission',r['summary']['secondary_joint_metric'],'wall',r['summary']['seconds'],flush=True)
 return i,arm
def snapshot():
 rows=[r for i in range(len(plan['cases'])) if (r:=validate_case(i)) is not None]
 result=dict(phase='RUNNING',completed_arms=sum((path(i,a)/'complete.json').exists() for i in range(len(plan['cases'])) for a in plan['arms']),total_arms=24,paired_cases=len(rows),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_updates=0,rows=rows)
 temporary=OUT/'progress.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(OUT/'progress.json')
 return rows
guard();snapshot()
with ThreadPoolExecutor(max_workers=plan['workers']) as ex:
 for f in as_completed([ex.submit(run,i,arm) for i in range(len(plan['cases'])) for arm in plan['arms']]):f.result();snapshot()
rows=snapshot();assert len(rows)==12
# Recompute from complete physical-clock traces, independent of saved summaries.
sys.path.insert(0,str(ROOT/'scripts'))
from gppo_joint_completion_metric import joint_completion_metric
for i,row in enumerate(rows):
 for arm in plan['arms']:
  folder=path(i,arm)/('rule' if arm=='rule' else 'gppo')
  trace=read(folder/'physical_joint_trace.json');assert len(trace)==3000
  metric=joint_completion_metric(trace,300,physics_dt=.1)
  assert metric==row['arms'][arm]['joint']
def score(rr):
 base=sum(r['arms']['rule']['joint']['secondary_joint_completion_seconds'] for r in rr)
 cand=sum(r['arms']['round8']['joint']['secondary_joint_completion_seconds'] for r in rr)
 eligible=all(r['arms'][a]['valid'] and r['arms'][a]['joint']['joint_success'] and r['arms'][a]['joint']['whole_window_safe'] and r['arms'][a]['nonrelay_complete'] and not r['arms'][a]['collisions'] for r in rr for a in plan['arms'])
 return dict(cases=len(rr),rule_mean_seconds=base/len(rr),round8_mean_seconds=cand/len(rr),completed_time_reduction=1-cand/base,inverse_time_efficiency_gain=base/cand-1,all_pairs_eligible=eligible)
pooled=score(rows)
groups={f"{n}_{'dynamic' if d else 'static'}":score([r for r in rows if r['case']['uavs']==n and r['case']['dynamic']==d]) for n in [30,60] for d in [False,True]}
seed_results={str(s):score([r for r in rows if r['case']['seed']==s]) for s in plan['reserved_final_seeds']}
# All27 ordered3-seed cluster bootstrap resamples, no Monte-Carlo error.
boot=[]
for ss in itertools.product(plan['reserved_final_seeds'],repeat=3):
 rr=[r for s in ss for r in rows if r['case']['seed']==s];boot.append(score(rr)['completed_time_reduction'])
import numpy as np
ci=np.quantile(boot,[.025,.975]).tolist()
passed=pooled['all_pairs_eligible'] and pooled['completed_time_reduction']>=.15 and all(g['completed_time_reduction']>=0 for g in groups.values())
guard()
result=dict(status='COMPLETE',verdict='PASS' if passed else 'FAIL',threshold=.15,primary=pooled,groups=groups,seed_results=seed_results,seed_cluster_bootstrap_95pct=ci,confidence_limitation='Only3 independent seeds; CI descriptive, not evidence that population lower bound exceeds15%',rows=rows,runs=24,paired_identity_verified=True,source_and_checkpoint_unchanged=True,protocol_sha256=protocol_hash,optimizer_updates=0,training_permitted=False,scope=plan['scope'],full_original_500m_goal_claim=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=plan['head'],git_status_after=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
(OUT/'acceptance_complete.json').write_text(json.dumps(result,indent=2))
print('INDEPENDENT_ACCEPTANCE_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['rows','git_status_after']},ensure_ascii=False),flush=True)
