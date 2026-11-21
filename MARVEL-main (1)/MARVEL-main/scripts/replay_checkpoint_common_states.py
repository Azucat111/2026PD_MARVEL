"""Matched-public-state policy replay; no optimizer, no new physical rollout."""
from pathlib import Path
import sys,json,hashlib,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.adapter import GPPOAssignment
OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
plan=json.loads((OUT/'plan.json').read_text());torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
models={}
specs=dict(plan['models'])
specs['round18']=dict(path='artifacts/extended_gppo_joint_batch_20261012_attempt18/checkpoint_batch_031.pth',edge_dim=6,sha256='f014db7958c80b7698ec422055af54a1e6ef5f749d149877108cfc36a4eb1fab')
specs['round19_initial']=dict(path='artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth',edge_dim=9,sha256='d3781e9817b1757628b48ecd8c55f6ad13a4cb8a312d58e7b52083671de14d88')
specs['round19_zero_opportunity']={**specs['round19'],'zero_opportunity':True}
for label,column in [('nominal_eta',6),('removal_regret',7),('context_flag',8)]:specs['round19_zero_'+label]={**specs['round19'],'zero_columns':[column]}
for name,spec in specs.items():
 path=ROOT/spec['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256'],name
 cp=torch.load(path,map_location='cpu',weights_only=False)
 with torch.random.fork_rng(devices=[]):m=GPPOActorCritic(edge_feature_dim=spec['edge_dim'])
 m.load_state_dict(cp['model']);m.eval();models[name]=m
inputs=[]
for folder in sorted((OUT/('pilots' if a.pilot else 'data')).glob('case_*/*')):
 if not (folder/'complete.json').exists():continue
 done=json.loads((folder/'complete.json').read_text())
 f=folder/('rule' if done['arm']=='rule' else 'gppo')/'actor_inputs_9.json'
 for j,row in enumerate(json.loads(f.read_text())):inputs.append((folder.relative_to(OUT).as_posix(),done['arm'],j,row))
rows=[];own_verified=0;parity_max=0.;parity_choices=0
for origin,arm,index,r in inputs:
 batch={k:torch.tensor(v) for k,v in r['batch'].items()}
 predictions={};logits={}
 for name,m in models.items():
  b=dict(batch);b['edge_features']=b['edge_features'][...,:specs[name]['edge_dim']].contiguous()
  if specs[name].get('zero_opportunity'):
   b['edge_features']=b['edge_features'].clone();b['edge_features'][...,6:]=0.
  if specs[name].get('zero_columns'):
   b['edge_features']=b['edge_features'].clone();b['edge_features'][...,specs[name]['zero_columns']]=0.
  with torch.no_grad():o=m(**b)
  assert bool(o['has_valid_action'][0]);z=o['flat_logits'][0].numpy();order=np.argsort(-z,kind='stable');action=int(order[0]);ti,ui=divmod(action,len(r['uav_ids']))
  legal=(~batch['action_mask'][0]).reshape(-1).numpy();legal_logits=z[legal]
  pvec=torch.softmax(torch.tensor(legal_logits),dim=0).numpy()
  predictions[name]=dict(action=action,uid=r['uav_ids'][ui],task=r['task_ids'][ti],logit_margin=float(z[order[0]]-z[order[1]]) if len(legal_logits)>1 else None,entropy=float(-(pvec*np.log(np.maximum(pvec,1e-30))).sum()),observed_total_seconds=float(r['raw_edge_seconds'][ti][ui][2]))
  logits[name]=z
 if arm in models:
  assert predictions[arm]['action']==r['action'],(origin,index,arm)
  own_verified+=1
 maxerr=float(np.max(np.abs(logits['round18']-logits['round19_initial'])));parity_max=max(parity_max,maxerr)
 parity_choices+=predictions['round18']['action']!=predictions['round19_initial']['action']
 rows.append(dict(origin=origin,index=index,time=r['time'],role='Search' if 100000<=r['chosen']['subtask_id']<200000 else 'Relay',actual=r['chosen'],predictions=predictions,public_state_sha256=hashlib.sha256(json.dumps(r['batch'],sort_keys=True).encode()).hexdigest()))
counts={}
for left,right in [('round8','round17'),('round8','round19'),('round17','round18'),('round18','round19'),('round19_initial','round19'),('round19','round19_zero_opportunity'),('round17','round19_zero_opportunity')]:
 rr=[r for r in rows if r['role']=='Search']
 counts[left+'__'+right]=dict(shared_search_states=len(rr),different_actions=sum(r['predictions'][left]['action']!=r['predictions'][right]['action'] for r in rr))
r=dict(states=len(rows),own_action_reproductions_verified=own_verified,pilot_only=a.pilot,comparisons=counts,round18_to_round19_zero_column_initialization=dict(max_logit_error=parity_max,different_greedy_choices=parity_choices),rows=rows,model_sha256={k:v['sha256'] for k,v in specs.items()},ablation_description='All zero-labelled predictions use identical final19 weights and recorded public state, changing only named edge columns to zero; diagnostic only, never physically deployed or trained.',optimizer_updates=0,causal_claim=False,limitation='Matched-state disagreement isolates policy differences but does not prove which alternative is physically faster; full closed-loop suite and bounded interventions are separate')
dest=OUT/('pilot_matched_state_replay.json' if a.pilot else f"matched_state_replay_{len(inputs):05d}.json")
dest.write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k not in ['rows','model_sha256']}),flush=True)
