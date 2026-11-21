"""Offline audit of frozen donor-filter activity. No policy/physics execution."""
from pathlib import Path
import json,datetime,collections,hashlib
O=Path(__file__).resolve().parent
D=O/'evaluation'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def filter_stats(events,arm):
 stats=dict(calls=len(events),expected_no_filter_calls=arm!='donor',no_filter_call_check=(len(events)==0 if arm!='donor' else None),applied_calls=0,missing_public_calls=0,fallback_calls=0,actual_mask_changed_calls=0,additional_masked_entries=0,unmasked_entries=0,blocked_ids_nonempty_calls=0,reason_counts=dict(collections.Counter(x.get('reason','UNSPECIFIED') for x in events)),malformed_mask_calls=[])
 for i,x in enumerate(events):
  stats['applied_calls']+=int(x.get('applied') is True);stats['missing_public_calls']+=int(bool(x.get('missing_public_position_ids')));stats['fallback_calls']+=int(x.get('fallback') is True);stats['blocked_ids_nonempty_calls']+=int(bool(x.get('blocked_ids')))
  a=x.get('original_mask');b=x.get('filtered_mask')
  if not isinstance(a,list) or not isinstance(b,list) or len(a)!=len(b) or any(not isinstance(r,list) for r in a+b) or any(len(r)!=len(s) for r,s in zip(a,b)):
   stats['malformed_mask_calls'].append(i);continue
  stats['actual_mask_changed_calls']+=int(a!=b)
  for oldrow,newrow in zip(a,b):
   for old,new in zip(oldrow,newrow):
    stats['additional_masked_entries']+=int(not bool(old) and bool(new));stats['unmasked_entries']+=int(bool(old) and not bool(new))
 return stats

def choices(records):
 result=[]
 for ordinal,r in enumerate(records):
  u=r['uav_ids'];t=r['task_ids'];action=int(r['action']);ti,ui=divmod(action,len(u));assert 0<=ti<len(t)
  result.append(dict(ordinal=ordinal,time=r['time'],step=r.get('step'),task_id=t[ti],uav_id=u[ui]))
 return result

def compare_ordered(unfiltered,donor,unfiltered_trace,donor_trace):
 a=choices(unfiltered);b=choices(donor)
 strict=lambda records:[{k:r.get(k) for k in ['time','step','uav_ids','task_ids','action']} for r in records]
 contexts_equal=strict(unfiltered)==strict(donor);masks_equal=(len(unfiltered)==len(donor) and all(x['batch']['action_mask']==y['batch']['action_mask'] for x,y in zip(unfiltered,donor)))
 first=None
 for i in range(max(len(a),len(b))):
  x=a[i] if i<len(a) else None;y=b[i] if i<len(b) else None
  if x!=y:first=dict(ordinal=i,unfiltered=x,donor=y);break
 return dict(comparison_method='Whole saved sequence in original order; semantic chosen task/UAV plus time/step; strict graph-ID/action-context equality reported separately. Never match by timestamp, never force probability replay.',unfiltered_records=len(a),donor_records=len(b),semantic_action_sequence_identical=a==b,strict_context_and_action_sequence_identical=contexts_equal,actual_input_mask_sequence_identical=masks_equal,first_semantic_difference=first,joint_metric_sample_trace_identical=unfiltered_trace==donor_trace,joint_trace_samples=dict(unfiltered=len(unfiltered_trace),donor=len(donor_trace)),joint_trace_hashes=dict(unfiltered=digest(unfiltered_trace),donor=digest(donor_trace)))

def load_arm(index,case,arm):
 folder=D/'data'/f"case_{index:02d}_{case['seed']}_{case['uavs']}_{'dynamic' if case['dynamic'] else 'static'}"/(arm+'_original')
 base=dict(arm=arm,status='PENDING',complete_marker=False)
 if not (folder/'complete.json').exists():return base,None
 try:
  complete=json.loads((folder/'complete.json').read_text());data=folder/('rule' if arm=='rule' else 'gppo');events=json.loads((data/'donor_filter_decisions.json').read_text());records=json.loads((data/'actor_inputs_9.json').read_text());trace=json.loads((data/'physical_joint_trace.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError) as e:return dict(base,status='INCOMPLETE_ARTIFACT',complete_marker=True,error=str(e)),None
 if complete.get('case')!=case or complete.get('optimizer_updates')!=0 or complete.get('source_stable') is not True or len(trace)!=3000 or not all(isinstance(x,list) for x in [events,records,trace]):return dict(base,status='INVALID_ARTIFACT',complete_marker=True,error='Case/update/source/list/fullH300 gate mismatch'),None
 stats=filter_stats(events,arm);status='COMPLETE' if not stats['malformed_mask_calls'] and stats['unmasked_entries']==0 and (arm=='donor' or stats['no_filter_call_check']) else 'FILTER_AUDIT_VIOLATION'
 return dict(base,status=status,complete_marker=True,filter=stats,actor_records=len(records),joint_trace_samples=len(trace)),dict(events=events,records=records,trace=trace)

def main():
 P=json.loads((D/'plan.json').read_text());rows=[]
 assert hashlib.sha256((D/'plan.json').read_bytes()).hexdigest()==(D/'protocol_sha256.txt').read_text().strip()
 for i,case in enumerate(P['cases']):
  arms={};payload={}
  for arm in P['arms']:arms[arm],payload[arm]=load_arm(i,case,arm)
  pair=dict(status='PENDING_COMPLETE_UNFILTERED_AND_DONOR')
  if payload.get('unfiltered') is not None and payload.get('donor') is not None:
   try:
    u=payload['unfiltered'];d=payload['donor'];pair=dict(status='COMPLETE',**compare_ordered(u['records'],d['records'],u['trace'],d['trace']))
    changes=arms['donor']['filter']['actual_mask_changed_calls']
    if changes==0:pair['interpretation_hint']='No actual donor mask change recorded; do not describe this case as evidence of a changed-action constraint effect.'
    elif pair['semantic_action_sequence_identical']:pair['interpretation_hint']='Donor mask changed but the observed chosen sequence stayed the same; altered excluded alternatives did not change recorded actions.'
    else:pair['interpretation_hint']='Donor mask changed and chosen sequence differs; benefit or harm still requires paired rawtrace completion/safety metrics.'
   except (KeyError,ValueError,AssertionError,ZeroDivisionError) as e:pair=dict(status='INVALID_ACTION_SEQUENCE',error=str(e))
  rows.append(dict(case_index=i,case=case,arms=arms,unfiltered_vs_donor=pair))
 complete=all(all(a['status']=='COMPLETE' for a in r['arms'].values()) and r['unfiltered_vs_donor']['status']=='COMPLETE' for r in rows)
 result=dict(status='COMPLETE' if complete else 'PARTIAL',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline donor-filter activity and ordered saved-action comparison only; no probability replay, training, physical execution, model selection or15percent claim.',plan_sha256=hashlib.sha256((D/'plan.json').read_bytes()).hexdigest(),expected_arms=len(P['cases'])*len(P['arms']),complete_valid_arms=sum(a['status']=='COMPLETE' for r in rows for a in r['arms'].values()),rows=rows,limits=['A joint metric trace contains radio/completion/safety-clock samples, not full robot motion; equality is not general trajectory or policy equivalence.','applied=True and blocked_ids nonempty are not the same as actual mask modification; fallbacks restore the original mask.','Changed actions do not establish improved completion; root rawtrace verifier remains authoritative.','Same-time allocation records are compared by original ordinal order, never joined by time.'])
 (O/'filter_activity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
