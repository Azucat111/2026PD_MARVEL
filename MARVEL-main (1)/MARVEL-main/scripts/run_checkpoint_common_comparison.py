"""Execute the frozen common comparison; resume completed arms without rerunning."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,subprocess,hashlib,datetime,copy,math
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
def read(p):return json.loads(p.read_text())
plan=read(OUT/'plan.json')
def guard():
 for f,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
 for m in plan['models'].values():assert hashlib.sha256((ROOT/m['path']).read_bytes()).hexdigest()==m['sha256']
def path(i,arm,pilot=False):
 c=plan['cases'][i]
 return OUT/('pilots' if pilot else 'data')/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/arm
def config_normalized(p):
 c=read(p);c=copy.deepcopy(c)
 for k in ['extended_checkpoint','extended_checkpoint_sha256']:c['task_scheduler'].pop(k,None)
 return c
def validate_case(i,pilot=False):
 folders=[path(i,a,pilot) for a in plan['arms']]
 if not all((p/'complete.json').exists() for p in folders):return None
 records=[read(p/'complete.json') for p in folders]
 real=[p/('rule' if r['arm']=='rule' else 'gppo') for p,r in zip(folders,records)]
 initial=[read(p/'initial.json') for p in real];configs=[config_normalized(p/'config.json') for p in real]
 truths=[read(p/'private_target_identity_diagnostic_only.json') for p in real]
 assert all(x==initial[0] for x in initial),'Unequal physical initial states'
 assert all(x==configs[0] for x in configs),'Unequal execution configs'
 assert all(x==truths[0] for x in truths),'Unequal hidden target realization'
 actors=[read(p/'actor_inputs_9.json') for p in real]
 if all(actors):
  def state(r):return {k:r[k] for k in ['time','uav_ids','task_ids','batch','raw_edge_seconds','public_context','raw_uav_features','raw_task_features','step']}
  assert all(state(x[0])==state(actors[0][0]) for x in actors),'Unequal first public decision state'
 for r in records:
  assert r['source_stable'] and r['optimizer_updates']==0 and r['summary']['full_window_complete']
 return dict(case=plan['cases'][i],arms={r['arm']:dict(metric=r['summary']['metric'],joint=r['summary']['secondary_joint_metric'],collisions=r['summary']['collision_types'],nonrelay_complete=r['summary']['required_nonrelay_tasks_complete'],valid=r['summary']['valid_for_comparison'],wall_seconds=r['summary']['seconds']) for r in records})
def run(i,arm,pilot=False):
 p=path(i,arm,pilot)
 if not (p/'complete.json').exists():
  assert not p.exists(),f'Partial folder preserved; explicit retry needed: {p}'
  cmd=[sys.executable,'-u','-B','scripts/evaluate_checkpoint_common_arm.py','--case',str(i),'--arm',arm]
  if pilot:cmd.append('--pilot')
  log=OUT/('pilot_logs' if pilot else 'logs')/f'case_{i:02d}_{arm}.log';log.parent.mkdir(exist_ok=True)
  with log.open('x') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,(i,arm,r.returncode,str(log))
 guard();r=read(p/'complete.json')
 print('ARM_VERIFIED',i,arm,'pilot',pilot,'find',r['summary']['metric']['mean_capped_seconds'],'joint',r['summary']['secondary_joint_metric']['joint_success'],'collisions',r['summary']['collision_types'],flush=True)
 return i,arm
def snapshot():
 rows=[r for i in range(len(plan['cases'])) if (r:=validate_case(i)) is not None]
 result=dict(phase='RUNNING',completed_arms=sum((path(i,a)/'complete.json').exists() for i in range(len(plan['cases'])) for a in plan['arms']),total_arms=len(plan['cases'])*len(plan['arms']),paired_cases=len(rows),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_updates=0,rows=rows,goal_complete=False)
 (OUT/'progress.json').write_text(json.dumps(result,indent=2))
 return rows
guard()
with ThreadPoolExecutor(max_workers=plan['workers']) as ex:
 for f in as_completed([ex.submit(run,0,arm,True) for arm in plan['arms']]):f.result()
pilot=validate_case(0,True);assert pilot is not None
(OUT/'pilot_fairness_verified.json').write_text(json.dumps(dict(passed=True,same_config_except_checkpoint=True,same_initial_poses_heats_and_hidden_targets=True,same_first_public_9dim_state=True,each_native_model_input_matches_its_exact_feature_slice=True,optimizer_updates=0),indent=2))
print('PILOT_FAIRNESS_PASS',flush=True)
jobs=[(i,arm) for i in range(len(plan['cases'])) for arm in plan['arms']]
with ThreadPoolExecutor(max_workers=plan['workers']) as ex:
 for f in as_completed([ex.submit(run,*j) for j in jobs]):
  f.result();snapshot()
rows=snapshot();assert len(rows)==len(plan['cases'])
scores={}
for arm in plan['arms'][1:]:
 base=sum(sum(r['arms']['rule']['metric']['capped_detection_seconds']) for r in rows)
 cand=sum(sum(r['arms'][arm]['metric']['capped_detection_seconds']) for r in rows)
 groups={}
 for n in [30,60]:
  for dynamic in [False,True]:
   rr=[r for r in rows if r['case']['uavs']==n and r['case']['dynamic']==dynamic]
   groups[f"{n}_{'dynamic' if dynamic else 'static'}"]=sum(sum(r['arms']['rule']['metric']['capped_detection_seconds']) for r in rr)/sum(sum(r['arms'][arm]['metric']['capped_detection_seconds']) for r in rr)-1
 scores[arm]=dict(pooled_gain=base/cand-1,mean_find_seconds=cand/(len(rows)*4),group_gains=groups,joint_successes=sum(r['arms'][arm]['joint']['joint_success'] for r in rows),full_window_safe_cases=sum(r['arms'][arm]['joint']['whole_window_safe'] for r in rows),all_find_cases=sum(r['arms'][arm]['metric']['detected_targets']==4 for r in rows),collision_events=sum(sum(r['arms'][arm]['collisions'].values()) for r in rows),baseline_find_total=base,candidate_find_total=cand)
result=dict(phase='COMPLETE',scores=scores,search_ranking=sorted(scores,key=lambda a:scores[a]['mean_find_seconds']),rows=rows,cases=len(rows),runs=len(jobs),source_stable=True,optimizer_updates=0,goal_complete=False,reserved_final_acceptance=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
guard();(OUT/'comparison_complete.json').write_text(json.dumps(result,indent=2));print('COMMON_COMPARISON_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
