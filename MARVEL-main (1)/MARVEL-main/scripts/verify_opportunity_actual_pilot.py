from pathlib import Path
import json,sys,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.gppo_model import GPPOActorCritic
oldfile=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18/checkpoint_batch_031.pth'
newfile=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth'
old=GPPOActorCritic(edge_feature_dim=6).eval();new=GPPOActorCritic(edge_feature_dim=9).eval()
old.load_state_dict(torch.load(oldfile,map_location='cpu',weights_only=False)['model']);new.load_state_dict(torch.load(newfile,map_location='cpu',weights_only=False)['model']);torch.set_num_threads(1)
folder=ROOT/'artifacts/extended_public_opportunity_natural_pilot_63090_20261013'
proof=json.loads((folder/'rollout_only_execution_proof.json').read_text());assert proof['passed'] and proof['optimizer_updates']==0
records=json.loads((folder/'episode_000/actor_inputs.json').read_text());worst=0.
for rec in records:
 batch={k:torch.as_tensor(v) for k,v in rec['batch'].items()};legacy=dict(batch);legacy['edge_features']=batch['edge_features'][...,:6]
 with torch.no_grad():a=old(**legacy);b=new(**batch)
 for k in ['flat_logits','value']:
  finite=torch.isfinite(a[k])&torch.isfinite(b[k]);assert torch.equal(torch.isfinite(a[k]),torch.isfinite(b[k]));error=float((a[k][finite]-b[k][finite]).abs().max()) if finite.any() else 0.;worst=max(worst,error)
  torch.testing.assert_close(a[k],b[k],atol=1e-5,rtol=1e-6)
 assert int(a['flat_logits'][0].argmax())==int(b['flat_logits'][0].argmax())
old.double();new.double();worst64=0.
for rec in records:
 batch={k:torch.as_tensor(v) for k,v in rec['batch'].items()};batch={k:(v.double() if v.is_floating_point() else v) for k,v in batch.items()};legacy=dict(batch);legacy['edge_features']=batch['edge_features'][...,:6]
 with torch.no_grad():a=old(**legacy);b=new(**batch)
 for k in ['flat_logits','value']:
  finite=torch.isfinite(a[k])&torch.isfinite(b[k]);error=float((a[k][finite]-b[k][finite]).abs().max()) if finite.any() else 0.;worst64=max(worst64,error);assert error<1e-10,(k,error)
summary=json.loads((folder/'episode_000/gppo/summary.json').read_text())
result=dict(passed=True,records=len(records),max_actor_and_critic_error=worst,max_double_precision_error=worst64,fp32_atol=1e-5,fp32_rtol=1e-6,all_argmax_equal=True,actual_public_feature_audit=proof['max_heading_feature_error'],pilot_optimizer_updates=0,find=summary['metric']['detected_targets'],collisions=summary['collision_types'],joint_acceptance_computed=False,acceptance_claim=False)
(ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/actual_natural_pilot_proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
