from pathlib import Path
import json,time,datetime,subprocess,sys,os,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];cut=datetime.datetime.fromisoformat('2026-11-11T06:15:00+00:00')
contract=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch_before_utc=cut.isoformat(),gate='All nine fixed training-case arms complete, common inputs matched, all arms success and fullH300safe; no gain-based selection',scope='Two prespecified new seeds, three arms, diagnostic not overall15percent acceptance',candidate_selection=False)
(O/'transfer_launch_contract.json').write_text(json.dumps(contract,indent=2));(O/'transfer_waiter_identity.json').write_text(json.dumps(dict(pid=os.getpid(),start=contract['created_utc'])))
while True:
 if datetime.datetime.now(datetime.timezone.utc)>=cut:
  (O/'transfer_launch_result.json').write_text(json.dumps(dict(status='NOT_STARTED_BUDGET')));raise SystemExit()
 f=O/'evaluation/execution.json';s=O/'evaluation/summary.json'
 if f.exists() and s.exists():break
 time.sleep(2)
summary=json.loads(s.read_text());good=summary['valid_for_comparison'] and all(r.get('success') and r.get('safe') for r in summary['rows'])
if not good:
 (O/'transfer_launch_result.json').write_text(json.dumps(dict(status='NOT_STARTED_TRAINING_CASE_GATE')));raise SystemExit()
for script in ['transfer/prepare_evaluation.py','transfer/run_evaluation.py']:
 print('TRANSFER_STAGE',script,flush=True);code=subprocess.call([sys.executable,'-u','-B',str(O/script)],cwd=R)
 if code:raise RuntimeError(f'{script} failed exit{code}')
(O/'transfer_launch_result.json').write_text(json.dumps(dict(status='FINISHED',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
