"""Re-use existing variable-graph collation and verify real 9D actor inputs."""
from pathlib import Path
import sys,json,ast,hashlib
import numpy as np
import torch
from typing import Dict,Sequence,List,Optional,Tuple
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.gppo_model import GPPOActorCritic
old=Path('/home/nick/MARVEL/utils/gppo_ppo.py');tree=ast.parse(old.read_text());names={'collate_task_graphs','remap_action_to_padded'}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(functions)==2
ns=dict(torch=torch,np=np,TaskGraph=object,Dict=Dict,List=List,Optional=Optional,Sequence=Sequence,Tuple=Tuple)
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(old),'exec'),ns)
class StoredGraph:
 def __init__(self,record):
  self.raw={k:torch.tensor(v)[0] for k,v in record['batch'].items() if k not in ['uav_valid_mask','task_valid_mask']}
  self.num_uavs=len(record['uav_ids']);self.num_subtasks=len(record['task_ids'])
  for k in ['uav_features','task_features','edge_features']:setattr(self,k,self.raw[k].numpy())
 def to_torch(self,device=None):return {k:v.to(device) for k,v in self.raw.items()}
 def unflatten_action(self,a):return divmod(a,self.num_uavs)
f=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19';records=[]
for i in range(8):
 r=json.loads((f/f'episode_{i:03d}/actor_inputs.json').read_text());records.extend([r[0],r[-1]])
graphs=[StoredGraph(r) for r in records];batch=ns['collate_task_graphs'](graphs,device='cpu')
assert batch['edge_features'].shape[-1]==9
m=GPPOActorCritic(edge_feature_dim=9);m.load_state_dict(torch.load(f/'checkpoint_batch_031.pth',map_location='cpu',weights_only=False)['model']);m.eval();torch.set_num_threads(1)
maxlog=0.;maxvalue=0.
with torch.no_grad():
 out=m(**batch)
 for i,(g,r) in enumerate(zip(graphs,records)):
  one=m(**{k:torch.tensor(v) for k,v in r['batch'].items()});a=ns['remap_action_to_padded'](g,r['action'],batch['uav_features'].shape[1])
  assert not batch['action_mask'][i].flatten()[a]
  x=torch.distributions.Categorical(logits=out['flat_logits'][i]/2.5).log_prob(torch.tensor(a))
  y=torch.distributions.Categorical(logits=one['flat_logits'][0]/2.5).log_prob(torch.tensor(r['action']))
  maxlog=max(maxlog,float(abs(x-y)));maxvalue=max(maxvalue,float(abs(out['value'][i]-one['value'][0])))
assert maxlog<1e-4 and maxvalue<1e-6
report=dict(passed=True,records=len(records),mixed_uav_counts=sorted({g.num_uavs for g in graphs}),mixed_task_counts=sorted({g.num_subtasks for g in graphs}),edge_dims=9,max_log_probability_error=maxlog,max_value_error=maxvalue,source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),reused_functions=sorted(names),training_steps=0,acceptance_claim=False)
(ROOT/'artifacts/extended_system_review_20261014_attempt19/existing_collation_reuse_contract.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
