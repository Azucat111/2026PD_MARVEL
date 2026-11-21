from pathlib import Path
import json,datetime
O=Path(__file__).resolve().parent;r=[]
for i in range(8):
 f=O/f'worker_{i:02d}/collection_proof.json'
 if f.exists():
  x=json.loads(f.read_text());m=x['alignment'];item=dict(index=i,completion=m['completion_seconds'],success=m['success'],safe=m['whole_window_safe']);print('COLLECT_COMPLETE',item);r.append(item)
 else:
  log=O/f'worker_{i:02d}.log'
  if log.exists():
   lines=[s for s in log.read_text().splitlines() if s.startswith('gppo ')];print('COLLECT_ACTIVE',i,lines[-1] if lines else 'initializing')
  else:print('COLLECT_QUEUED',i)
for stage in ['evaluation','transfer']:
 f=O/stage/'summary.json'
 if f.exists():
  x=json.loads(f.read_text());print(stage,json.dumps(x)[:1600])
f=O/'guarded/update_proof.json'
if f.exists():
 x=json.loads(f.read_text());print('UPDATE',{k:v for k,v in x.items() if k in ['status','records','optimizer_steps','eligible_for_evaluation','mean_exact_kl','max_state_exact_kl','training_state_greedy_changes']})
(O/'live_status.json').write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_new_rollouts=len(r),collection=r),indent=2))

