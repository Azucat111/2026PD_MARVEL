from pathlib import Path
import json,time,datetime,subprocess,sys,hashlib,os
O=Path(__file__).resolve().parent;R=O.parents[1]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
review=json.loads((O/'update_independent_review.json').read_text());assert not review['important_defects_found'];assert sha(O/'update_once.py')==review['reviewed_update_source_sha256'];assert sha(O/'update_contract.json')==review['reviewed_update_contract_sha256']
(O/'pipeline_identity.json').write_text(json.dumps(dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
def run(script):
 print('START_STAGE',str(script),datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 code=subprocess.call([sys.executable,'-u','-B',str(O/script)],cwd=R)
 if code:raise RuntimeError(f'{script} failed exit{code}')
 print('END_STAGE',str(script),flush=True)
while not (O/'execution.json').exists():
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-11-11T05:34:00+00:00'):raise TimeoutError('Collection budget exhausted')
 time.sleep(2)
rows=json.loads((O/'execution.json').read_text());assert len(rows)==8 and all(r.get('exit_code')==0 for r in rows)
run(Path('update_once.py'));u=json.loads((O/'update_proof.json').read_text());assert u['eligible_for_evaluation']
run(Path('evaluation/prepare_evaluation.py'));run(Path('evaluation/run_evaluation.py'))
print('TRAINING_CASE_PIPELINE_COMPLETE',flush=True)
