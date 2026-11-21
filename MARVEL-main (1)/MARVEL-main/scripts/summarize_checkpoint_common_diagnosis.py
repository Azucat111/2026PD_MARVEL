"""Finalize common-suite evidence and per-target explanations, without training."""
from pathlib import Path
import json,hashlib,itertools,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
def read(p):return json.loads(p.read_text())
plan=read(OUT/'plan.json');r=read(OUT/'comparison_complete.json');causal=read(OUT/'causal_complete.json')
for file,sha in plan['source_hashes'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
for m in plan['models'].values():assert hashlib.sha256((ROOT/m['path']).read_bytes()).hexdigest()==m['sha256']
assert hashlib.sha256(Path(plan['frozen_checkpoint']).read_bytes()).hexdigest()==plan['frozen_sha256']
replay=read(sorted(OUT.glob('matched_state_replay_*.json'))[-1]);details=[];radio={a:[] for a in plan['arms']}
for i,row in enumerate(r['rows']):
 c=row['case'];d=OUT/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"
 arms={};initial_physical_states=[]
 for arm in plan['arms']:
  f=d/arm/('rule' if arm=='rule' else 'gppo');events=read(f/'events.json');inputs=read(f/'actor_inputs_9.json');tr=read(f/'physical_joint_trace.json')
  with (f/'mission_frames.jsonl').open() as frames:initial_physical_states.append(json.loads(next(frames))['physical_poses_diagnostic_only'])
  ts=row['arms'][arm]['metric']['capped_detection_seconds'];found={e['data']['target_index']:dict(detector_uid=e['data']['robot_id'],sensor_step=e['step'],mode=e['data']['detection_mode']) for e in events if e['type']=='target_found'}
  assignments={}
  for q in inputs:
   tid=q['chosen']['subtask_id']
   if 100000<=tid<200000:assignments.setdefault(tid-100000,[]).append(dict(time=q['time'],uid=q['chosen']['uav_id']))
  arms[arm]=dict(find_seconds=ts,find_events=found,search_assignments=assignments,first_joint_seconds=row['arms'][arm]['joint']['secondary_joint_completion_seconds'],joint_success=row['arms'][arm]['joint']['joint_success'],final_radio=tr[-1]['connectivity'],final_3s_radio_stable=all(z['connectivity']>=.9 for z in tr[-31:]),safe=row['arms'][arm]['joint']['whole_window_safe'])
  radio[arm].append(dict(case_index=i,final_radio=arms[arm]['final_radio'],final3s=arms[arm]['final_3s_radio_stable']))
 assert all(z==initial_physical_states[0] for z in initial_physical_states),'Unequal initial physical positions/headings'
 details.append(dict(case_index=i,case=c,initial_physical_poses_and_headings_equal=True,arms=arms))
for arm in plan['arms'][1:]:
 r['scores'][arm]['mean_joint_completion_seconds_failures_capped_300']=float(np.mean([row['arms'][arm]['joint']['secondary_joint_completion_seconds'] for row in r['rows']]))
 r['scores'][arm]['final_radio_stable_cases']=sum(z['final3s'] for z in radio[arm])
winner=r['search_ranking'][0];pairwise={}
for arm in plan['arms'][1:]:
 if arm==winner:continue
 best_totals=[sum(row['arms'][winner]['metric']['capped_detection_seconds']) for row in r['rows']]
 alt_totals=[sum(row['arms'][arm]['metric']['capped_detection_seconds']) for row in r['rows']]
 seed_values=[]
 for seed in sorted({c['seed'] for c in plan['cases']}):
  idx=[i for i,c in enumerate(plan['cases']) if c['seed']==seed]
  seed_values.append((sum(alt_totals[i] for i in idx),sum(best_totals[i] for i in idx)))
 boot=[sum(seed_values[i][0] for i in ids)/sum(seed_values[i][1] for i in ids)-1 for ids in itertools.product(range(len(seed_values)),repeat=len(seed_values))]
 pairwise[arm]=dict(best_speed_ratio_gain=sum(alt_totals)/sum(best_totals)-1,best_faster_cases=sum(a>b+1e-8 for a,b in zip(alt_totals,best_totals)),equal_cases=sum(abs(a-b)<=1e-8 for a,b in zip(alt_totals,best_totals)),best_slower_cases=sum(a<b-1e-8 for a,b in zip(alt_totals,best_totals)),three_seed_block_bootstrap_95_percentile=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],uncertainty_limit='Only three historical seeds, not independent final acceptance')
losses=[]
for d in details:
 for arm in plan['arms'][1:]:
  if arm==winner:continue
  for target,(old,best) in enumerate(zip(d['arms'][arm]['find_seconds'],d['arms'][winner]['find_seconds'])):
   if old>best+1e-8:losses.append(dict(case=d['case'],case_index=d['case_index'],model=arm,best_model=winner,target=target,loss_seconds=old-best,model_find=old,best_find=best,model_assignments=d['arms'][arm]['search_assignments'].get(str(target),d['arms'][arm]['search_assignments'].get(target,[])),best_assignments=d['arms'][winner]['search_assignments'].get(str(target),d['arms'][winner]['search_assignments'].get(target,[]))))
losses.sort(key=lambda x:-x['loss_seconds'])
for probe in causal['results']:
 c=probe['case'];origin=f"data/case_{probe['case_index']:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}/{probe['original_arm']}"
 match=next(z for z in replay['rows'] if z['origin']==origin and z['index']==probe['record_index'])
 probe['same_state_stage_choices']={k:dict(task=v['task'],uid=v['uid']) for k,v in match['predictions'].items()}
 probe['round19_initial_and_final_choose_same_action']=match['predictions']['round19_initial']['action']==match['predictions']['round19']['action']
 probe['round17_and_round18_choose_same_action']=match['predictions']['round17']['action']==match['predictions']['round18']['action']
eligible=[arm for arm in r['search_ranking'] if r['scores'][arm]['joint_successes']==len(plan['cases']) and r['scores'][arm]['full_window_safe_cases']==len(plan['cases'])]
result=dict(status='COMPLETE',scores=r['scores'],search_best=winner,search_ranking=r['search_ranking'],best_with_all_common_joint_cases=eligible[0] if eligible else None,paired_comparisons=pairwise,radio_diagnostics=radio,largest_per_target_losses=losses[:24],per_case_details=details,causal_probes=causal['results'],feature_schema_audit=read(OUT/'feature_schema_audit.json'),matched_state_replay=dict(states=replay['states'],own_action_reproductions_verified=replay['own_action_reproductions_verified'],comparisons=replay['comparisons'],round18_to_round19_initialization=replay['round18_to_round19_zero_column_initialization']),training_update_audit={k:v for k,v in read(OUT/'training19_update_direction_audit.json').items() if k not in ['records_detail','batches_detail']},source_and_weights_unchanged=True,optimizer_updates=0,comparison_runs=len(plan['cases'])*4,reserved_final_seeds_used=False,project_goal_complete=False,scope='Same current execution conditions on 3 historical seeds x30/60 xstatic/dynamic H300. Diagnostic ranking only; not original500m or heldout15% acceptance.',causal_limits='A local one-action improvement proves that allocation mattered in those cases. It does not uniquely identify a PPO hyperparameter or prove one global training failure mechanism.')
(OUT/'final_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k in ['status','scores','search_best','search_ranking','best_with_all_common_joint_cases','paired_comparisons','comparison_runs','source_and_weights_unchanged','optimizer_updates']}),flush=True)
