"""Final read-only verification and original-goal alignment; no score replacement."""
from pathlib import Path
import json,hashlib,datetime,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
def read(p):return json.loads(p.read_text())
plan=read(OUT/'plan.json');comparison=read(OUT/'comparison_complete.json');final=read(OUT/'final_diagnosis.json')
assert comparison['runs']==48 and len(comparison['rows'])==12
for f,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
for m in plan['models'].values():assert hashlib.sha256((ROOT/m['path']).read_bytes()).hexdigest()==m['sha256']
assert hashlib.sha256(Path(plan['frozen_checkpoint']).read_bytes()).hexdigest()==plan['frozen_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['head']
mission={}
for arm in plan['arms']:
 rows=[r['arms'][arm] for r in comparison['rows']]
 mean=sum(r['joint']['secondary_joint_completion_seconds'] for r in rows)/len(rows)
 assert all(r['joint']['joint_success'] and r['joint']['whole_window_safe'] and not r['collisions'] for r in rows)
 mission[arm]=dict(mean_completed_joint_mission_seconds=mean,cases=len(rows),all_completed=True,all_full_h300_safe=True)
baseline=mission['rule']['mean_completed_joint_mission_seconds']
for arm in plan['arms'][1:]:
 m=mission[arm];m['completed_mission_time_reduction']=1-m['mean_completed_joint_mission_seconds']/baseline;m['inverse_time_efficiency_gain']=baseline/m['mean_completed_joint_mission_seconds']-1
mission_best=min(plan['arms'][1:],key=lambda a:mission[a]['mean_completed_joint_mission_seconds'])
# Supplementary joint first-control allocation probe, separate from original48.
probe=OUT/'joint_prefix_probes/case_02_round19_to_round17';original=OUT/'data/case_02_9164_60_static/round19'
for f in ['config.json','initial.json','private_target_identity_diagnostic_only.json']:
 assert read(probe/'gppo'/f)==read(original/'gppo'/f),f
pr=read(probe/'complete.json');ps=pr['summary']
assert ps['metric']['detected_targets']==4 and not ps['collision_types'] and pr['source_stable'] and pr['optimizer_updates']==0
assert len(ps['causal_intervention'])==2 and all(x['step']==10 for x in ps['causal_intervention'])
original_summary=read(original/'gppo/summary.json');joint_probe=dict(case=pr['case'],original_mean_find=original_summary['metric']['mean_capped_seconds'],intervention_mean_find=ps['metric']['mean_capped_seconds'],intervention_find_times=ps['metric']['capped_detection_seconds'],override_count=2,override_control_step=10,all_later_decisions='original round19',collision_events=0,ranking_unchanged=True,full_joint_acceptance_claim=False,optimizer_updates=0)
# Original goal quotes verified from the user-provided historical handoff.
handoff=Path('/mnt/c/Users/ASUS/Downloads/MARVEL_GPPO_PROJECT_CONTEXT_HANDOFF_2026-09-21.md')
text=handoff.read_text(encoding='utf-8')
assert '多机任务效率提升 **15%+**' in text and 'paired-completed mission time' in text
audit=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_handoff=str(handoff),handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),original_goal_quote='相比规则调度，多机任务效率提升15%+; paired-completed mission time',discovered_metric_misalignment=True,search_find_was_used_as_primary_in_recent_training_evaluation=True,original_48_run_search_ranking_retained=comparison['search_ranking'],original_goal_mission_time_ranking=sorted(plan['arms'][1:],key=lambda a:mission[a]['mean_completed_joint_mission_seconds']),mission_metrics=mission,project_goal_best_on_this_historical_suite=mission_best,search_specialist_on_this_suite=final['search_best'],posthoc_original_goal_alignment_review=True,not_preregistered_reserved_final_acceptance=True,goal_complete=False,reserved_seeds_used=False,training_permitted=False,optimizer_updates=0,scope='Historical3 seeds x30/60 xstatic/dynamic, same connected80m domain H300; all48 genuinely completed and safe. Mission ranking is a disclosed original-goal alignment audit, not a retroactive claim of final15% acceptance.',joint_prefix_causal_probe=joint_probe)
(OUT/'mission_goal_alignment_audit.json').write_text(json.dumps(audit,indent=2))
backup=OUT/'final_diagnosis_search_frozen_before_goal_audit.json'
assert not backup.exists();backup.write_bytes((OUT/'final_diagnosis.json').read_bytes())
final['original_goal_alignment_audit']='mission_goal_alignment_audit.json';final['project_goal_best_on_this_historical_suite']=mission_best;final['mission_metrics']=mission;final['supplementary_joint_prefix_probe']=joint_probe;final['original_goal_metric_misalignment_detected']=True;final['project_goal_complete']=False;final['training_permitted']=False
(OUT/'final_diagnosis.json').write_text(json.dumps(final,indent=2))
progress_path=ROOT/'artifacts/extended_training_execution_progress_20261010.json';progress=read(progress_path);progress['checkpoint_common_comparison_complete']=True;progress['active_bounded_probe']=None;progress['training_permitted']=False;progress['goal_metric_misalignment_detected']=True;progress['original_mission_goal_candidate']=mission_best;progress['search_specialist_candidate']=final['search_best'];progress['project_goal_complete']=False;progress['status']='Fixed48 comparison COMPLETE; original mission goal alignment clarified: round8 best joint completion, round17 best Search; all training remains paused';progress_path.write_text(json.dumps(progress,indent=2))
(OUT/'task_verification_complete.json').write_text(json.dumps(dict(status='COMPLETE',comparison_runs=48,all_source_and_weights_unchanged=True,all48_full_h300_complete_safe=True,original_goal_best=mission_best,search_best=final['search_best'],optimizer_updates=0,reserved_final_acceptance=False,head=plan['head'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
print(json.dumps(dict(status='COMPLETE',mission_metrics=mission,mission_best=mission_best,search_best=final['search_best'],joint_prefix_probe=joint_probe,optimizer_updates=0,goal_complete=False),ensure_ascii=False),flush=True)
