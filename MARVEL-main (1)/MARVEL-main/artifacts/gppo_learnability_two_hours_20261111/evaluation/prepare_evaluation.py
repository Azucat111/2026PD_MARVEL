"""Freeze one fixed diagnostic evaluation; never select or train models."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;O=D.parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 if (D/'plan.json').exists():raise FileExistsError('Diagnostic plan already frozen; refuse overwrite')
 parent=R/'artifacts/gppo_three_hour_20261110/training/batch_00/behavior.pth'
 candidate=O/'guarded/update/checkpoint_batch_000.pth'
 if not candidate.is_file():raise FileNotFoundError('Diagnostic candidate does not exist: '+str(candidate))
 proof_path=O/'guarded/update_proof.json'
 if not proof_path.is_file():raise FileNotFoundError('Require diagnostic update gate: '+str(proof_path))
 proof=json.loads(proof_path.read_text())
 eligible=proof.get('eligible_for_evaluation',proof.get('eligible_for_validation',False))
 if proof.get('status')!='COMPLETE' or eligible is not True:raise ValueError('Diagnostic update gate rejected or incomplete')
 if proof.get('candidate_sha256') and proof['candidate_sha256']!=sha(candidate):raise ValueError('Update gate candidate hash mismatch')
 if proof.get('candidate') and (R/proof['candidate']).resolve()!=candidate.resolve():raise ValueError('Update gate points to another candidate')
 base=json.loads((R/'artifacts/gppo_three_hour_20261110/heldout/plan.json').read_text())
 for rel,digest in base['source_hashes'].items():
  if sha(R/rel)!=digest:raise ValueError('Prior shared executor/assets changed: '+rel)
 models={'round8':base['models']['round8'],
  'parent':dict(path=str(parent.relative_to(R)),sha256=sha(parent),edge_dim=6),
  'candidate':dict(path=str(candidate.relative_to(R)),sha256=sha(candidate),edge_dim=6)}
 sources=dict(base['source_hashes'])
 for name in ['evaluate_arm.py','prepare_evaluation.py','run_evaluation.py','summarize_evaluation.py']:
  path=D/name;sources[str(path.relative_to(R))]=sha(path)
 for path in sorted((O/'guarded').glob('*')):
  if path.is_file() and path.suffix in ('.py','.json') and path.name not in ('update_proof.json','independent_review.json'):
   sources[str(path.relative_to(R))]=sha(path)
 sources[str((D/'protocol_amendment.json').relative_to(R))]=sha(D/'protocol_amendment.json')
 cases=[{'seed': 1880021, 'uavs': 30, 'dynamic': False}, {'seed': 1280022, 'uavs': 30, 'dynamic': True}, {'seed': 1880024, 'uavs': 60, 'dynamic': True}]
 plan={key:base[key] for key in ['head','branch','git_root','scenario','warning_rate_mps','frozen_checkpoint','frozen_sha256']}
 plan.update(scope='Three training cases: learnability diagnostic only, not independent acceptance.',
  diagnostic_training_cases=True,independent_diagnostic=False,independent_acceptance=False,acceptance_claim=False,
  training_permitted=False,model_selection_permitted=False,selection='none; fixed parent and single gated diagnostic candidate',
  optimizer_updates=0,cases=cases,arms=['rule','parent','candidate'],models=models,source_hashes=sources,
  horizon_seconds=300,physics_steps=3000,physics_dt=.1,whole_window_safety_required=True,
  wall_budget_seconds_per_arm=1500,runtime_deadline_utc='2026-11-11T06:20:00+00:00',
  parent_deadline_utc='2026-11-11T06:21:00+00:00',worker_limit=4,
  jobs=[dict(case=i,arm=arm) for i in range(3) for arm in ['rule','parent','candidate']],
  update_proof_path=str(proof_path.relative_to(R)),update_proof_sha256=sha(proof_path),source_guard_count=len(sources),
  frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with (D/'plan.json').open('x') as stream:json.dump(plan,stream,indent=2)
 (D/'protocol_sha256.txt').write_text(sha(D/'plan.json'))
 print('TRAINING_CASE_DIAGNOSTIC_FROZEN',len(sources),'source guards; no independent acceptance or model selection')
if __name__=='__main__':main()
