"""One constrained Ridge fit: exact paired labels plus censor inequalities."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from paired_gain_calibration import fit_bounded_gain_model,candidate_representations,select_gain_action
from paired_latency_intervals import latency_interval,efficiency_interval
OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20';plan=json.loads((OUT/'plan.json').read_text());index=json.loads((OUT/'selected_branches.json').read_text())
assert json.loads((OUT/'collection_complete_recovered.json').read_text())['complete'];assert not (OUT/'candidate.pth').exists()
torch.set_num_threads(1)
def folder(seed,rank):
 p=(OUT/index[f'{seed}:{rank}']).resolve();p.relative_to(OUT.resolve());return p
def state(seed):return json.loads((folder(seed,0)/'first_decision.json').read_text())
def actions_batch(f):
 g=f['graph'];u=g['uav_ids'];t=g['task_ids'];a=[t.index(o['task'])*len(u)+u.index(o['uid']) for o in f['top_public_options']]
 return {k:torch.tensor(v) for k,v in f['actor_batch'].items()},a
def outcomes(seed):
 rows=[json.loads((folder(seed,k)/'complete.json').read_text()) for k in range(4)]
 assert all(r['full_short_window'] and not r['policy_failures'] for r in rows)
 return rows,[latency_interval(r['metric']) for r in rows]
# Source audit: horizon affects action masks through deadline screening; none of
# the recorded calls changed masks through that path, so these prefixes retain
# the same public decisions when the declared final horizon is enlarged.
deadlines=0
for p in index.values():
 for d in json.loads((OUT/p/'rule/decisions.json').read_text()):deadlines+=len(d['deadline_masks'])
assert deadlines==0,'Recollect matching H300 prefixes if actual deadline masks differ'
warm=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth';cp=torch.load(warm,map_location='cpu',weights_only=False);m=GPPOActorCritic(edge_feature_dim=9);m.load_state_dict(cp['model']);m.eval()
x=[];y=[];cx=[];cu=[];covered=set();counts=[]
for i,seed in enumerate(plan['training_seeds']):
 rows,labels=outcomes(seed);b,a=actions_batch(state(seed));z=candidate_representations(m,b,a);delta=z-z[0]
 baseline=labels[0];exact=0;bounded=0
 if not baseline['exact']:continue
 for rank,r in enumerate(labels):
  if rows[rank]['collision_types']:continue
  if r['exact']:x.append(delta[rank]);y.append(baseline['lower']-r['lower']);exact+=1
  else:cx.append(delta[rank]);cu.append(baseline['lower']-r['lower']);bounded+=1
 if exact>1 or bounded:
  case=plan['case_cycle'][i%4];covered.add((case['uavs'],bool(case['dynamic'])))
 counts.append(dict(seed=seed,exact=exact,censored_upper_constraints=bounded))
assert covered=={(30,False),(60,False),(30,True),(60,True)}
started=time.perf_counter();reg=fit_bounded_gain_model(np.asarray(x),np.asarray(y),np.asarray(cx).reshape(-1,128),np.asarray(cu),alpha=10.);fit_seconds=time.perf_counter()-started
old=cp['model']
with torch.no_grad():m.actor.scorer[-1].weight.copy_(torch.tensor(reg.coef_,dtype=torch.float32).reshape(1,-1));m.actor.scorer[-1].bias.zero_()
changed=[k for k,v in m.state_dict().items() if not torch.equal(v,old[k])];assert set(changed)<={'actor.scorer.4.weight','actor.scorer.4.bias'}
cp=dict(cp);cp['model']=m.state_dict();cp['training_stage']='paired bounded gain calibration';cp['paired_calibration']=dict(decision_scope='first Search only; top4 legal public choices; remaining strong rule',minimum_gain_seconds=.05,fit_calls=1,alpha=10.,training_seeds=plan['training_seeds'],exact_labels=len(y),censored_constraints=len(cu),fit_seconds=fit_seconds)
p=OUT/'candidate.pth';torch.save(cp,p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
# Freeze before reading any held-out outcomes. Predictions use only actor inputs.
predictions=[]
for seed in plan['validation_seeds']:
 b,a=actions_batch(state(seed))
 with torch.no_grad():scores=m(**b)['flat_logits'][0].numpy()
 action=select_gain_action(scores,b['action_mask'][0].numpy(),a,.05);rank=a.index(action)
 predictions.append(dict(seed=seed,chosen_rank=rank,predicted_gain=float(scores[action]-scores[a[0]])))
(OUT/'candidate_frozen_before_validation.json').write_text(json.dumps(dict(sha256=sha,fit_calls=1,fit_seconds=fit_seconds,training=counts,training_domains=sorted(covered),exact_labels=len(y),interval_constraints=len(cu),changed_parameters=changed,predictions=predictions,validation_labels_seen=False,actual_deadline_vetoes_in_recorded_prefixes=deadlines,acceptance_claim=False),indent=2))
validation=[];bs=[];cs=[]
for pred in predictions:
 rows,bounds=outcomes(pred['seed']);rank=pred['chosen_rank'];baseline=bounds[0];candidate=bounds[rank];bs.append(baseline);cs.append(candidate)
 validation.append(dict(**pred,baseline=baseline,candidate=candidate,collisions=sum(rows[rank]['collision_types'].values())))
gain=efficiency_interval(bs,cs);eligible=all(r['candidate']['exact'] and r['candidate']['found']==4 and r['collisions']==0 for r in validation) and gain['lower']>=.15
assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
result=dict(candidate_sha256=sha,fit_calls=1,fit_seconds=fit_seconds,validation=validation,efficiency_gain_bounds=gain,eligible_for_h300_development=eligible,project_goal_complete=False,acceptance_claim=False,limitations='Censor bounds are not imputed Find times. H20 search prefixes do not certify full H300 joint safety/radio lifecycle.')
(OUT/'validation_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
