from pathlib import Path
import sys,json,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.heading_features import HEADING_PROTOCOL
folder=ROOT/'artifacts/extended_integrated_natural_61206_30_dynamic_20261011_attempt1';manifest=json.loads((folder/'manifest.json').read_text());cp=Path(manifest['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
checkpoint=torch.load(cp,map_location='cuda',weights_only=False);assert checkpoint['protocol_version']==HEADING_PROTOCOL
model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(checkpoint['model'],strict=True);model.eval();torch.set_num_threads(1)
records=json.loads((folder/'gppo/actor_inputs.json').read_text());maxerror=0.
for record in records:
 batch={k:torch.as_tensor(v,device='cuda') for k,v in record['batch'].items()}
 with torch.no_grad():output=model(**batch)
 assert int(output['flat_logits'][0].argmax())==record['action']
 assert not bool(batch['action_mask'].reshape(-1)[record['action']])
 maxerror=max(maxerror,abs(float(output['value'][0])-record['value']))
assert records and maxerror<1e-6
result=dict(production_recorded_decisions=len(records),all_masked_argmax_replayed=True,max_value_error=maxerror,candidate_sha256=manifest['checkpoint_sha256'],passed=True,acceptance_claim=False)
with (folder/'production_inference_replay.json').open('x') as s:json.dump(result,s,indent=2)
print(json.dumps(result))
