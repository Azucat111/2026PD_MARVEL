from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'artifacts/extended_goal_certificate_cycle_61166_live_observed_20261010_attempt4/manifest.json').read_text())
stable=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['source_hashes'].items())
old=json.loads((root/'artifacts/extended_gppo_joint_batch_20261010_attempt6/manifest.json').read_text())
changes=[p for p,h in old['source_hashes'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
allowed={'integrations/gppo/extended_travel.py','integrations/gppo/scheduler.py','integrations/gppo/search_recovery.py'}
assert set(changes)==allowed
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
sha=hashlib.sha256(frozen.read_bytes()).hexdigest();assert sha=='57abf685f7a8226dcf71e1441333bf74da6e228e423f1bb0034e110975e3fcd9'
assert stable
result=dict(current_fixture_source_stable=stable,changed_since_interrupted_training=changes,frozen_checkpoint_sha256=sha,git_root=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),passed=True,acceptance_claim=False)
with (root/'artifacts/extended_goal_route_certificate_20261010_attempt1/source_guard.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
