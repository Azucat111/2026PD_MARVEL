from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from integrations.gppo.opportunity_features import initialize_opportunity_model,OPPORTUNITY_PROTOCOL
from integrations.gppo.gppo_model import GPPOActorCritic
source=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18/checkpoint_batch_031.pth'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f014db7958c80b7698ec422055af54a1e6ef5f749d149877108cfc36a4eb1fab'
cp=torch.load(source,map_location='cpu',weights_only=False);model=initialize_opportunity_model(cp['model'],'cpu')
# Retain critic and reward horizon; reset only optimizer for expanded parameters.
result=dict(cp);result.pop('optimizer',None);result['model']=model.state_dict();result['protocol_version']=OPPORTUNITY_PROTOCOL
result['warm_start_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();result['input_expansion_zero_initialized']=True
out=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth';assert not out.exists();torch.save(result,out)
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),native_unchanged=True)))
