"""Summarize completed fixed diagnostic arms; never choose a checkpoint."""
from pathlib import Path
import json,hashlib,datetime
D=Path(__file__).resolve().parent

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def summarize(write=True):
 P=json.loads((D/'plan.json').read_text());rows=[];pairs=[]
 for i,case in enumerate(P['cases']):
  group=[]
  for arm in P['arms']:
   folder=D/'data'/f"case_{i:02d}_{case['seed']}_{case['uavs']}_{'dynamic' if case['dynamic'] else 'static'}"/(arm+'_original')
   row=dict(case=i,**case,arm=arm,status='PENDING')
   try:
    complete=json.loads((folder/'complete.json').read_text());s=complete['summary'];m=s['secondary_joint_metric'];sub=folder/('rule' if arm=='rule' else 'gppo')
    initial=json.loads((sub/'initial.json').read_text());hidden=json.loads((sub/'private_target_identity_diagnostic_only.json').read_text());config=json.loads((sub/'config.json').read_text())
    for key in ['extended_checkpoint','extended_checkpoint_sha256']:config.get('task_scheduler',{}).pop(key,None)
    row.update(status='COMPLETE',seconds=m['secondary_joint_completion_seconds'],success=m['joint_success'],safe=m['whole_window_safe'],mean_find=s['metric']['mean_capped_seconds'],initial_sha256=digest(initial),hidden_layout_sha256=digest(hidden),normalized_config_sha256=digest(config),source_stable=complete['source_stable'],wall_seconds=s['seconds'])
    group.append(row)
   except (FileNotFoundError,json.JSONDecodeError):pass
   if row['status']=='PENDING':
    try:
     execution=json.loads((D/f'case_{i}_{arm}_execution.json').read_text())
     if execution.get('status')=='NOT_STARTED_BUDGET':row['status']='NOT_STARTED_BUDGET'
     elif execution.get('exit_code') is not None:row['status']='FAILED' if execution['exit_code']!=0 else 'MISSING_COMPLETE'
    except (FileNotFoundError,json.JSONDecodeError):pass
   rows.append(row)
  checks={key:len({r[key] for r in group})==1 if len(group)>=2 else None for key in ['initial_sha256','hidden_layout_sha256','normalized_config_sha256']}
  pairs.append(dict(case=i,**case,completed_arms=len(group),common_condition_checks=checks,all_conditions_match=all(checks.values()) if len(group)>=2 else None,valid_for_comparison=(len(group)==len(P['arms']) and all(checks.values()) and all(r['source_stable'] for r in group))))
 result=dict(status='COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL',scope=P['scope'],diagnostic_training_cases=P['diagnostic_training_cases'],independent_diagnostic=P['independent_diagnostic'],independent_acceptance=False,model_selection_permitted=False,selection='none',completed_arms=sum(r['status']=='COMPLETE' for r in rows),expected_arms=len(rows),rows=rows,cases=pairs,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 result['valid_for_comparison']=result['status']=='COMPLETE' and all(case['valid_for_comparison'] for case in pairs)
 if write:
  import os
  temp=D/f'.summary_{os.getpid()}.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(D/'summary.json')
 return result
if __name__=='__main__':print(json.dumps(summarize(),indent=2))
