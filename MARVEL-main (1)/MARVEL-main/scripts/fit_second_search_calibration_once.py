"""Conditional one-fit second-action head; never consumes validation outcomes."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from paired_gain_calibration import candidate_representations,fit_bounded_gain_model,select_gain_action
from paired_latency_intervals import latency_interval
OUT=ROOT/'artifacts/extended_second_search_preflight_20261015_attempt21'
plan=json.loads((OUT/'plan.json').read_text());ceiling=json.loads((OUT/'scope_ceiling_complete.json').read_text())
assert ceiling['eligible_for_low_cost_fit'] and not (OUT/'candidate.pth').exists()
torch.set_num_threads(1)
FIRST=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/candidate.pth'
assert hashlib.sha256(FIRST.read_bytes()).hexdigest()==plan['first_candidate_sha256']
cp=torch.load(FIRST,map_location='cpu',weights_only=False);m=GPPOActorCritic(edge_feature_dim=9);m.load_state_dict(cp['model']);m.eval()
def folder(seed,rank):
 base=OUT/'data'/f'seed_{seed}_rank_{rank}'
 rows=[p for p in [base,base.with_name(base.name+'_retry_01')] if (p/'complete.json').exists()]
 assert len(rows)==1
 return rows[0]
x=[];y=[];cx=[];cu=[];states=[]
for seed in plan['training_seeds']:
 d=json.loads((folder(seed,0)/'search_decisions.json').read_text())[1];g=d['graph']
 batch={k:torch.tensor(v) for k,v in d['actor_batch'].items()}
 actions=[g['task_ids'].index(o['task'])*len(g['uav_ids'])+g['uav_ids'].index(o['uid']) for o in d['top_public_options']]
 z=candidate_representations(m,batch,actions);delta=z-z[0]
 rows=[json.loads((folder(seed,k)/'complete.json').read_text()) for k in range(4)]
 bounds=[latency_interval(r['metric']) for r in rows]
 assert bounds[0]['exact']
 for rank,b in enumerate(bounds):
  if rows[rank]['collision_types']:continue
  if b['exact']:x.append(delta[rank]);y.append(bounds[0]['lower']-b['lower'])
  else:cx.append(delta[rank]);cu.append(bounds[0]['lower']-b['lower'])
 states.append((seed,batch,actions))
started=time.perf_counter();reg=fit_bounded_gain_model(np.asarray(x),np.asarray(y),np.asarray(cx).reshape(-1,128),np.asarray(cu),alpha=10.);elapsed=time.perf_counter()-started
old={k:v.clone() for k,v in cp['model'].items()}
with torch.no_grad():
 m.actor.scorer[-1].weight.copy_(torch.tensor(reg.coef_,dtype=torch.float32).reshape(1,-1));m.actor.scorer[-1].bias.zero_()
changed=[k for k,v in m.state_dict().items() if not torch.equal(v,old[k])]
assert set(changed)<={'actor.scorer.4.weight','actor.scorer.4.bias'}
cp=dict(cp);cp.pop('paired_calibration')
cp['model']=m.state_dict();cp['first_search_head']={k:old[k] for k in ['actor.scorer.4.weight','actor.scorer.4.bias']}
cp['two_stage_calibration']=dict(scope='first two Search only; independently frozen first head; top4 legal public choices; rest strong rule',minimum_gain_seconds=.05,first_source_sha256=plan['first_candidate_sha256'],fit_calls=1,fit_seconds=elapsed,alpha=10,training_seeds=plan['training_seeds'])
cp['training_stage']='two-stage paired head calibration; sampler only until validation'
p=OUT/'candidate.pth';torch.save(cp,p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
predictions=[]
for seed,b,a in states:
 with torch.no_grad():scores=m(**b)['flat_logits'][0].numpy()
 choice=select_gain_action(scores,b['action_mask'][0].numpy(),a,.05)
 predictions.append(dict(seed=seed,second_rank=a.index(choice)))
result=dict(candidate_sha256=sha,first_candidate_sha256=plan['first_candidate_sha256'],fit_calls=1,fit_seconds=elapsed,changed_parameters=changed,exact_labels=len(y),censor_constraints=len(cu),training_predictions=predictions,validation_labels_seen=False,goal_complete=False,acceptance_claim=False)
(OUT/'candidate_frozen_before_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
