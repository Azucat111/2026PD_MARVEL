"""One fixed-cost fit using paired training labels; seal model before validation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from paired_gain_calibration import fit_gain_model,paired_gain,candidate_representations,select_gain_action
OUT=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20';plan=json.loads((OUT/'plan.json').read_text());torch.set_num_threads(1)
assert json.loads((OUT/'collection_complete_recovered.json').read_text())['complete']
index=json.loads((OUT/'selected_branches.json').read_text())
def branch(seed,rank):
 path=(OUT/index[f'{seed}:{rank}']).resolve();path.relative_to(OUT.resolve());return path
assert not (OUT/'candidate.pth').exists(),'Never repeat the registered fit'
model=GPPOActorCritic(edge_feature_dim=9);warm=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth'
cp=torch.load(warm,map_location='cpu',weights_only=False);model.load_state_dict(cp['model']);model.eval()
def state(seed):
 return json.loads((branch(seed,0)/'first_decision.json').read_text())
def batch_actions(f):
 b={k:torch.tensor(v) for k,v in f['actor_batch'].items()};g=f['graph'];u=g['uav_ids'];t=g['task_ids']
 actions=[t.index(o['task'])*len(u)+u.index(o['uid']) for o in f['top_public_options']]
 return b,actions
def labels(seed):
 rows=[json.loads((branch(seed,rank)/'complete.json').read_text()) for rank in range(4)]
 assert all(r['full_short_window'] and not r['policy_failures'] for r in rows)
 exact=all(r['metric']['detected_targets']==4 and not r['collision_types'] for r in rows)
 return rows,exact
xs=[];ys=[];used=[];excluded=[]
for seed in plan['training_seeds']:
 rows,exact=labels(seed)
 if not exact:excluded.append(seed);continue
 b,actions=batch_actions(state(seed));z=candidate_representations(model,b,actions)
 baseline=rows[0]['metric']['mean_capped_seconds']
 xs.extend(z-z[0]);ys.extend(paired_gain(baseline,r['metric']['mean_capped_seconds']) for r in rows);used.append(seed)
covered=set()
for seed in used:
 case=plan['case_cycle'][plan['training_seeds'].index(seed)%4]
 covered.add((case['uavs'],bool(case['dynamic'])))
if covered!={(30,False),(60,False),(30,True),(60,True)}:
 (OUT/'fit_rejected_missing_domain.json').write_text(json.dumps(dict(reason='Exact paired training labels do not cover all four declared domains',covered=sorted(covered),usable_seeds=used,excluded_seeds=excluded,fit_calls=0,acceptance_claim=False),indent=2));raise SystemExit(2)
if len(used)<4:
 (OUT/'fit_rejected.json').write_text(json.dumps(dict(reason='Insufficient exact paired training labels',usable=used,excluded=excluded,training_updates=0,acceptance_claim=False),indent=2));raise SystemExit(2)
x=np.asarray(xs);y=np.asarray(ys);ridge=fit_gain_model(x,y,alpha=plan['training']['alpha'])
with torch.no_grad():
 model.actor.scorer[-1].weight.copy_(torch.tensor(ridge.coef_,dtype=torch.float32).reshape(1,-1));model.actor.scorer[-1].bias.zero_()
new_state=model.state_dict();old=cp['model'];changed=[k for k in old if not torch.equal(old[k],new_state[k])]
assert set(changed)<= {'actor.scorer.4.weight','actor.scorer.4.bias'}
cp=dict(cp);cp['model']=new_state;cp['training_stage']='paired gain calibration';cp['paired_calibration']=dict(fit_calls=1,alpha=10.,train_seeds=used,excluded_seeds=excluded,decision_scope='first Search only; top4 legal public choices; remaining strong rule',minimum_gain_seconds=.05)
path=OUT/'candidate.pth';torch.save(cp,path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'candidate_frozen_before_validation.json').write_text(json.dumps(dict(sha256=sha,fit_calls=1,training_seeds=used,excluded=excluded,labels=len(y),changed_parameters=changed,training_mae=float(np.mean(abs(ridge.predict(x)-y))),validation_labels_seen=False,acceptance_claim=False),indent=2))
# Only now open independently reserved short-validation outcomes.
validation=[]
for seed in plan['validation_seeds']:
 b,actions=batch_actions(state(seed))
 with torch.no_grad():scores=model(**b)['flat_logits'][0].numpy()
 choice=select_gain_action(scores,b['action_mask'][0].numpy(),actions,min_gain=.05)
 rank=actions.index(choice)
 rows,exact=labels(seed);base=rows[0]['metric'];selected=rows[rank]['metric']
 validation.append(dict(seed=seed,chosen_rank=rank,predicted_gain=float(scores[choice]-scores[actions[0]]),exact=exact,baseline_mean=base['mean_capped_seconds'],candidate_mean=selected['mean_capped_seconds'],found=selected['detected_targets'],collisions=sum(rows[rank]['collision_types'].values())))
gain=sum(r['baseline_mean'] for r in validation)/sum(r['candidate_mean'] for r in validation)-1
eligible=all(r['exact'] and r['found']==4 and r['collisions']==0 for r in validation) and gain>=.15
assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
result=dict(candidate_sha256=sha,fit_calls=1,validation=validation,pooled_gain=gain,eligible_for_h300_development=eligible,acceptance_claim=False,project_goal_complete=False,limits='One fitted fixed candidate, four independent H20 case seeds; cached physical branches validate first decision only, no H300 joint certificate')
(OUT/'validation_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
