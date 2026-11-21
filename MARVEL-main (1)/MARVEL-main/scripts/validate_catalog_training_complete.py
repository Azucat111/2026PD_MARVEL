from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1];folder=root/'artifacts/extended_gppo_joint_batch_20261010_attempt7'
manifest=json.loads((folder/'manifest.json').read_text());complete=json.loads((folder/'complete.json').read_text())
assert complete['episodes']==16 and complete['updates']==2 and complete['initial_unchanged']
assert manifest['args']['seeds']==list(range(61180,61196))
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['source_hashes'].items())
initial=root/manifest['args']['initial'];assert hashlib.sha256(initial.read_bytes()).hexdigest()==manifest['initial_sha256']
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth');assert hashlib.sha256(frozen.read_bytes()).hexdigest()==manifest['frozen_checkpoint_sha256']
cp=folder/'checkpoint_batch_001.pth';sha=hashlib.sha256(cp.read_bytes()).hexdigest()
result=dict(complete_episodes=16,optimizer_updates=2,source_stable=True,initial_unchanged=True,frozen_unchanged=True,final_checkpoint_sha256=sha,passed=True,acceptance_claim=False)
with (folder/'post_training_source_guard.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
