"""Freeze an evaluation-only common suite before observing outcomes."""
from pathlib import Path
import json,hashlib,subprocess,datetime,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
models={
 'round8':dict(path='artifacts/extended_gppo_joint_batch_20261010_attempt8/checkpoint_batch_001.pth',sha256='869e2ac49e54d24143c497197c27c39800b882fa1e4100b2f021815f73f30298',edge_dim=6),
 'round17':dict(path='artifacts/extended_gppo_joint_batch_20261012_attempt17/checkpoint_batch_015.pth',sha256='1cda34ecb47ec1eb411e3bf47f30351bbec1a7d48b96ff6297f70ac0739cf22a',edge_dim=6),
 'round19':dict(path='artifacts/extended_gppo_joint_batch_20261013_attempt19/checkpoint_batch_031.pth',sha256='7942984adcf3f55e77bac53b02291d0472fbc22f9b259b322395d4bbe9f20fb0',edge_dim=9)}
for m in models.values():assert sha(ROOT/m['path'])==m['sha256']
sources=list((ROOT/'integrations').rglob('*.py'))+list((ROOT/'utils').rglob('*.py'))
sources += [ROOT/p for p in ['parameter.py','test_parameter.py','scripts/benchmark_fair_search.py','scripts/gppo_benchmark_domains.py','scripts/gppo_joint_completion_metric.py','scripts/gppo_motion_capture.py','scripts/gppo_training_records.py','scripts/filter_scheduler_candidates.py','scripts/gppo_acceptance_protocol.py','configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml','configs/scenarios/assets/extended_connected_gppo_20261010.png','configs/scenarios/extended_live_gppo.yaml','configs/scenarios/assets/extended_live_gppo.png','configs/dynamics/kinematic_simple.yaml','configs/sensors/ideal_cached_extended_20261012.yaml','configs/sensors/ideal.yaml','configs/communication/ideal.yaml','load_model/MARVEL/checkpoint.pth']]
hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
for p in sources:
 if p.suffix in ['.py','.yaml','.png']:
  target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
cases=[dict(seed=s,uavs=n,dynamic=d) for s in [9164,9168,9170] for n in [30,60] for d in [False,True]]
plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_root=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status_before=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),models=models,cases=cases,arms=['rule','round8','round17','round19'],horizon_seconds=300,workers=4,wall_budget_seconds_per_arm=7200,scenario='configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml',warning_rate_mps=.2,source_hashes=hashes,frozen_checkpoint=str(frozen),frozen_sha256=sha(frozen),deterministic=True,python_hash_seed=0,training_permitted=False,optimizer_updates=0,source_policy='No production edits; official extended adapter loads native 6/9-dimensional checkpoints without conversion; common observed-information production executors and unchanged physical rules',randomness='Preserve Torch CPU/CUDA RNG around adapter construction/setup so model input dimensionality cannot perturb simulation random streams; runtime reset and policy greedy controls otherwise unchanged',scoring='Pooled sum(rule true sensor Find times capped at300)/sum(candidate times)-1. Report all12 and four scenario groups; no dropping failures. Joint criterion is full H300 zeroColl/NoFly, all nonrelay complete, physical radio>=.9 stable3s and Safety released.',ranking='Separate primary Search ranking from complete joint eligibility; do not promote latest automatically; do not claim reserved-seed acceptance',limitations='These three historical development seeds are deliberately common for diagnosis; not unseen generalization or original500m-map acceptance.',reserved_seeds_untouched=[101101,202202,303303],additional_causal_probe='After fixed suite, at most two largest paired Search losses may receive one first-divergence action replacement with the best checkpoint same-public-state greedy action, all later decisions original model; diagnostic only, not included in ranking')
assert plan['branch']=='extended-scalability'
(OUT/'plan.json').write_text(json.dumps(plan,indent=2));(OUT/'heartbeat.json').write_text(json.dumps(dict(phase='FROZEN_NOT_STARTED',optimizer_updates=0)))
print(json.dumps(dict(out=str(OUT),cases=len(cases),runs=len(cases)*4,source_files=len(hashes),free_gb=shutil.disk_usage(ROOT).free/1e9,models=models)),flush=True)
