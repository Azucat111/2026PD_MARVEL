"""Read checkpoint metadata and hash, never train or execute an optimizer."""
from pathlib import Path
import json,hashlib,torch
ROOT=Path(__file__).resolve().parents[1]
p=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
c=torch.load(p,map_location='cpu',weights_only=False)
result={'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'keys':list(c),'metadata':{}}
for key,value in c.items():
    if key in ('model','optimizer','scheduler','rng_states'):continue
    try:text=json.dumps(value)
    except (TypeError,ValueError):continue
    if len(text)<30000:result['metadata'][key]=value
out=ROOT/'artifacts/search_frozen_training_metadata_20261010.json'
assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
