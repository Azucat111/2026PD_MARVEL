"""Rank first; then at most two single-action causal probes. No ranking contamination."""
from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
plan=json.loads((OUT/'plan.json').read_text());comparison=json.loads((OUT/'comparison_complete.json').read_text())
assert comparison['source_stable'] and comparison['optimizer_updates']==0
def run_script(file,args,log):
 with (OUT/log).open('x') as f:r=subprocess.run([sys.executable,'-u','-B','scripts/'+file,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(file,r.returncode,log)
run_script('replay_checkpoint_common_states.py',[],'matched_state_replay.log')
winner=comparison['search_ranking'][0];original='round19' if winner!='round19' else comparison['search_ranking'][-1]
case_losses=[]
for i,row in enumerate(comparison['rows']):
 difference=sum(row['arms'][original]['metric']['capped_detection_seconds'])-sum(row['arms'][winner]['metric']['capped_detection_seconds'])
 if difference>1e-8:case_losses.append((difference,i,row))
case_losses.sort(reverse=True,key=lambda x:x[0]);jobs=[];selected=set()
for diff,i,row in case_losses:
 identity=(row['case']['seed'],row['case']['uavs'])
 if identity in selected:continue
 c=row['case'];folder=OUT/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/original/'gppo'
 records=json.loads((folder/'actor_inputs_9.json').read_text())
 # Replay artifact is indexed by folder and ordinal, with candidate-native inputs.
 replay_path=sorted(OUT.glob('matched_state_replay_*.json'))[-1];replay=json.loads(replay_path.read_text())
 origin=str(folder.parent.relative_to(OUT))
 rr=[r for r in replay['rows'] if r['origin']==origin]
 assert len(rr)==len(records)
 different=[r for r in rr if r['predictions'][winner]['action']!=r['predictions'][original]['action']]
 assignments={}
 with (folder/'mission_frames.jsonl').open() as frames:
  for line in frames:
   frame=json.loads(line);pairs={(100000+int(h),int(uid)) for h,uid in frame.get('public_search_assignments',{}).items()}
   pairs|={(200000+int(target),int(v['helper'])) for target,v in frame.get('public_relay_assignments',{}).items()}
   assignments[int(frame['step'])]=pairs
 # Changing only the order of the same two concurrent assignments is not
 # an effective policy intervention. Pick the first differing legal pair
 # absent from the original final assignment set at that control tick.
 differing=[r for r in different if (r['predictions'][winner]['task'],r['predictions'][winner]['uid']) not in assignments[records[r['index']]['step']]]
 if not differing:continue
 first=differing[0];pred=first['predictions'][winner];record=records[first['index']]
 jobs.append(dict(case_index=i,case=c,original_arm=original,alternative_arm=winner,record_index=first['index'],step=record['step'],public_state_sha256=first['public_state_sha256'],alternative_task_uid=[pred['task'],pred['uid']],baseline_task_uid=[record['chosen']['subtask_id'],record['chosen']['uav_id']],original_find_total=sum(row['arms'][original]['metric']['capped_detection_seconds']),winner_find_total=sum(row['arms'][winner]['metric']['capped_detection_seconds']),original_gap_seconds=diff))
 selected.add(identity)
 if len(jobs)==2:break
cp=dict(jobs=jobs,ranking_frozen_before_probes=comparison['search_ranking'],selection='Largest positive whole-case Search losses, at most two distinct seed/UAV-count groups; replace first differing legal allocation that changes the actual assignment set; skip pure allocation-order permutations, use best checkpoint on exact recorded public state, all later decisions original checkpoint',optimizer_updates=0,diagnostic_only=True,full_joint_claim=False,causal_runner_sha256=hashlib.sha256((ROOT/'scripts/evaluate_checkpoint_causal_probe.py').read_bytes()).hexdigest())
(OUT/'causal_plan.json').write_text(json.dumps(cp,indent=2))
results=[]
for job in jobs:
 i=job['case_index'];src=job['original_arm'];alt=job['alternative_arm']
 run_script('evaluate_checkpoint_causal_probe.py',['--case',str(i),'--arm',src,'--alternative',alt],f'causal_case_{i:02d}_{src}_to_{alt}.log')
 folder=OUT/'causal_probes'/f'case_{i:02d}_{src}_to_{alt}'
 result=json.loads((folder/'complete.json').read_text())
 c=job['case'];original_folder=OUT/'data'/f"case_{i:02d}_{c['seed']}_{c['uavs']}_{'dynamic' if c['dynamic'] else 'static'}"/src/'gppo'
 assert json.loads((folder/'gppo/initial.json').read_text())==json.loads((original_folder/'initial.json').read_text())
 assert json.loads((folder/'gppo/config.json').read_text())==json.loads((original_folder/'config.json').read_text())
 assert json.loads((folder/'gppo/private_target_identity_diagnostic_only.json').read_text())==json.loads((original_folder/'private_target_identity_diagnostic_only.json').read_text())
 total=sum(result['summary']['metric']['capped_detection_seconds']);recovered=job['original_find_total']-total
 results.append(dict(**job,intervention_find_total=total,recovered_find_seconds=recovered,recovered_fraction_of_original_gap=recovered/job['original_gap_seconds'],collision_types=result['summary']['collision_types'],intervention=result['summary']['causal_intervention'],all_find=result['summary']['metric']['detected_targets']==4))
(OUT/'causal_complete.json').write_text(json.dumps(dict(results=results,optimizer_updates=0,ranking_unchanged=True,interpretation='One allocator intervention changes future closed-loop trajectory; effects are local to selected case and action, not a general proof about all PPO updates'),indent=2))
print('CAUSAL_COMPLETE',json.dumps(results),flush=True)
