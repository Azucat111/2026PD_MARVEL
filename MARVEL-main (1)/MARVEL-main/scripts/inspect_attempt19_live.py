"""Read-only authoritative process and cohort progress snapshot."""
from pathlib import Path
import json,psutil
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19'
names={'scripts/resume_opportunity_attempt19.py','scripts/guard_opportunity_attempt19_detached.py','scripts/run_opportunity_rollout_worker.py','scripts/continue_completed_opportunity_attempt19.py','scripts/evaluate_opportunity_gppo_joint_acceptance.py'}
live=[]
for proc in psutil.process_iter(['pid','ppid','cmdline','status']):
 try:
  item=proc.info;args=item['cmdline'] or [];matched=names.intersection(args)
  if not matched or item['status']==psutil.STATUS_ZOMBIE:continue
  row=dict(pid=item['pid'],ppid=item['ppid'],status=item['status'],script=sorted(matched)[0])
  if '--out' in args:
   output=args[args.index('--out')+1];row['output']=output
   log=ROOT/(output+'.log')
   if log.exists():
    text=log.read_text();lines=[x for x in text.splitlines() if x.startswith('gppo ')];row['last_progress']=lines[-1] if lines else None;row['traceback']='Traceback' in text
  live.append(row)
 except (psutil.NoSuchProcess,psutil.AccessDenied,ProcessLookupError):continue
history=json.loads((folder/'progress.json').read_text());updates=json.loads((folder/'updates.json').read_text())
result=dict(episodes=len(history),updates=len(updates),all_found=sum(r['metric']['detected_targets']==4 for r in history),collisions=sum(sum(r['collision_types'].values()) for r in history),complete=(folder/'complete.json').exists(),development_complete=(folder/'development_9170_complete.json').exists(),live=live)
print(json.dumps(result,ensure_ascii=False))
