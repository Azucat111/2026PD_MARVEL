"""Training-only grouped cross-validation using sklearn/SciPy; never held-out labels."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from paired_gain_calibration import candidate_representations,fit_bounded_gain_model
OLD=ROOT/'artifacts/extended_second_search_preflight_20261015_attempt21'
OUT=ROOT/'artifacts/extended_safe_public_calibration_20261015_attempt22';OUT.mkdir(exist_ok=False)
plan=dict(training_seeds=list(range(63400,63408)),fresh_validation_seeds=list(range(63416,63420)),first_head='attempt20 frozen',feature_options=['trunk128','trunk128_plus_observed_edge9'],alphas=[.1,1.,10.],unsafe_gain_upper=-.05,selection='Grouped leave-one-seed-out, require zero selected training-fold collisions and >=15% pooled gain before any new fit/validation',heldout_validation_labels_allowed=False,expensive_training_allowed=False,goal_complete=False)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2))
torch.set_num_threads(1)
cp=torch.load(ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/candidate.pth',map_location='cpu',weights_only=False)
m=GPPOActorCritic(edge_feature_dim=9);m.load_state_dict(cp['model']);m.eval()
original=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20';oldindex=json.loads((original/'selected_branches.json').read_text())
def folder(seed,rank):
 base=OLD/'data'/f'seed_{seed}_rank_{rank}'
 p=[x for x in [base,base.with_name(base.name+'_retry_01')] if (x/'complete.json').exists()]
 assert len(p)==1
 return p[0]
data=[]
for seed in plan['training_seeds']:
 rec=json.loads((folder(seed,0)/'search_decisions.json').read_text())[1];g=rec['graph']
 batch={k:torch.tensor(v) for k,v in rec['actor_batch'].items()}
 acts=[g['task_ids'].index(o['task'])*len(g['uav_ids'])+g['uav_ids'].index(o['uid']) for o in rec['top_public_options']]
 trunk=candidate_representations(m,batch,acts)
 public=batch['edge_features'][0].reshape(-1,9)[acts].numpy().astype(float)
 rows=[json.loads((folder(seed,k)/'complete.json').read_text()) for k in range(4)]
 assert all(r['metric']['detected_targets']==4 for r in rows)
 means=np.array([r['metric']['mean_capped_seconds'] for r in rows]);unsafe=np.array([bool(r['collision_types']) for r in rows])
 assert not unsafe[0]
 base=json.loads((original/oldindex[f'{seed}:0']/'complete.json').read_text())['metric']['mean_capped_seconds']
 data.append(dict(seed=seed,trunk128=trunk-trunk[0],trunk128_plus_observed_edge9=np.concatenate([trunk-trunk[0],public-public[0]],axis=1),gains=means[0]-means,means=means,unsafe=unsafe,baseline=base))
results=[]
started=time.perf_counter()
for feature in plan['feature_options']:
 for alpha in plan['alphas']:
  predictions=[]
  for hold in data:
   train=[d for d in data if d['seed']!=hold['seed']]
   x=np.concatenate([d[feature][~d['unsafe']] for d in train]);y=np.concatenate([d['gains'][~d['unsafe']] for d in train])
   cx=np.concatenate([d[feature][d['unsafe']] for d in train]);cu=np.full(len(cx),plan['unsafe_gain_upper'])
   scaler=StandardScaler(with_mean=False).fit(np.concatenate([d[feature] for d in train]))
   reg=fit_bounded_gain_model(scaler.transform(x),y,scaler.transform(cx) if len(cx) else cx,cu,alpha=alpha)
   scores=reg.predict(scaler.transform(hold[feature]));rank=int(np.argmax(scores))
   if scores[rank]-scores[0]<.05:rank=0
   predictions.append(dict(seed=hold['seed'],rank=rank,mean=float(hold['means'][rank]),baseline=hold['baseline'],unsafe=bool(hold['unsafe'][rank])))
  gain=sum(r['baseline'] for r in predictions)/sum(r['mean'] for r in predictions)-1
  result=dict(feature=feature,alpha=alpha,pooled_gain=gain,unsafe_selected=sum(r['unsafe'] for r in predictions),predictions=predictions)
  results.append(result);print('GROUP_CV',json.dumps(result),flush=True)
eligible=[r for r in results if not r['unsafe_selected'] and r['pooled_gain']>=.15]
summary=dict(training_only=True,group_cv=results,eligible_for_final_low_cost_fit=bool(eligible),selected=max(eligible,key=lambda r:r['pooled_gain']) if eligible else None,seconds=time.perf_counter()-started,goal_complete=False,acceptance_claim=False,unsafe_training_labels=3,validation_labels_seen=False,expensive_training_allowed=False)
(OUT/'group_cv_complete.json').write_text(json.dumps(summary,indent=2));print('GROUP_CV_COMPLETE',json.dumps(summary),flush=True)
