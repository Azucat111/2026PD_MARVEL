"""Create a separate feasible benchmark map; never alter historical input maps."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import yaml
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'configs/scenarios/assets/extended_live_gppo.png'
scenario=ROOT/'configs/scenarios/extended_live_gppo.yaml'
target=ROOT/'configs/scenarios/assets/extended_connected_gppo_20261010.png'
yaml_target=ROOT/'configs/scenarios/extended_connected_gppo_20261010.yaml'
metadata=ROOT/'artifacts/extended_connected_map_generation_20261010.json'
if any(p.exists() for p in (target,yaml_target,metadata)):raise FileExistsError('Preserve all existing outputs; generator is one-shot')
source_hash=hashlib.sha256(source.read_bytes()).hexdigest();scenario_hash=hashlib.sha256(scenario.read_bytes()).hexdigest()
raw=np.array(Image.open(source).convert('L'));assert raw.shape==(80,80)
before=raw.copy();raw[34:59,12:20]=255
assert np.all(raw[:4,:]<128) and np.all(raw[-4:,:]<128) and np.all(raw[:,:4]<128) and np.all(raw[:,-4:]<128)
structure=np.array([[0,1,0],[1,1,1],[0,1,0]])
old_labels,old_count=label(before>=128,structure=structure);new_labels,new_count=label(raw>=128,structure=structure)
assert old_count==2 and new_count==1
Image.fromarray(raw).save(target)
config=yaml.safe_load(scenario.read_text())
config['scenario']['name']='extended_connected_gppo_20261010'
config['scenario']['description']='Separate feasibility benchmark: preserve the prior 80m static crop and add an 8m wide public map corridor x[12,20), y[34,59) connecting its two free regions. Historical map remains unchanged. UAV speed, radius, radio range, task capacities and frozen native protocol are unchanged. Both scheduling arms use this identical declared map.'
config['environment']['map_file']='assets/'+target.name
yaml_target.write_text(yaml.safe_dump(config,sort_keys=False))
metadata.write_text(json.dumps(dict(source=str(source),source_sha256=source_hash,original_scenario_sha256=scenario_hash,
    generated=str(target),generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    original_free_components=old_count,generated_free_components=new_count,modified_cells=int(np.count_nonzero(raw!=before)),
    original_free_cells=int(np.count_nonzero(before>=128)),generated_free_cells=int(np.count_nonzero(raw>=128)),
    modification={'x_half_open':[12,20],'y_half_open':[34,59],'cell_size_m':1},acceptance_claim=False),indent=2))
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
assert hashlib.sha256(scenario.read_bytes()).hexdigest()==scenario_hash
print(metadata.read_text())
