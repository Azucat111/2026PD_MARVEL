"""Offline action-value head for remaining GLOBAL capped sensor-Find cost."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,torch,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.gppo_model import GPPOActorCritic
OLD=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19'
DATA=ROOT/'artifacts/extended_offline_find_cost_head_20261015_attempt28_retry01'
OUT=ROOT/'artifacts/extended_global_find_advantage_20261015_attempt30';OUT.mkdir(exist_ok=False)
plan=dict(source='Existing917 strict matched public decisions; global label includes all4 historical capped true sensor Find times',target='Mean max(capped_Find_time - decision_time,0), minus cross-fitted state-only critic baseline',state_baseline='Frozen GPPO critic last-hidden representation plus action-independent public graph context; StandardScaler+Ridge alpha100; three train-block folds',actor_head='HistGradientBoostingRegressor; global cost advantage, greedy min within original highest-priority Search hard masks',parameters=dict(loss='squared_error',max_iter=150,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10.,learning_rate=.05,early_stopping=False,random_state=42),state_independence='No selected-pair features enter V(s)',primary_label_horizon=150,missing_target_semantics='Historical H150 capped objective, never imputed true Find',minimum_predicted_gain_seconds=.1,new_simulations=0,ppo_updates=0,expensive_training_allowed=False,goal_complete=False)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2));torch.set_num_threads(1)
a=np.load(DATA/'observed_training_arrays.npz');x=a['features'];hold=a['holdout'];episodes=a['episodes']
metadata=json.loads((DATA/'label_provenance.json').read_text())['labels']
base=joblib.load(DATA/'candidate.joblib');warm=Path(base['frozen_encoder'])
assert hashlib.sha256(warm.read_bytes()).hexdigest()==base['frozen_encoder_sha256']
cp=torch.load(warm,map_location='cpu',weights_only=False);m=GPPOActorCritic(edge_feature_dim=9);m.load_state_dict(cp['model']);m.eval()
records={};times={};zs=[];y=[]
for row in metadata:
 ep=row['episode']
 if ep not in records:
  records[ep]=json.loads((OLD/f'episode_{ep:03d}'/'actor_inputs.json').read_text())
  summary=json.loads((OLD/f'episode_{ep:03d}'/'gppo/summary.json').read_text());times[ep]=np.asarray(summary['metric']['capped_detection_seconds'],dtype=float)
 matches=[r for r in records[ep] if r['time']==row['time'] and r['chosen']['uav_id']==row['chosen_uid'] and r['chosen']['subtask_id']==100000+row['target']]
 assert len(matches)==1
 r=matches[0];batch={k:torch.tensor(v) for k,v in r['batch'].items()};captured=[]
 hook=m.critic.value[-1].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy()[0].copy()))
 try:
  with torch.no_grad():m(**batch)
 finally:hook.remove()
 zs.append(captured[0]);y.append(float(np.maximum(times[ep]-row['time'],0.).mean()))
z=np.asarray(zs);y=np.asarray(y)
state=np.concatenate([z,x[:,[40,41,42,43,45,46]]],axis=1)
assert state.shape==(len(x),134)
def value_model():return make_pipeline(StandardScaler(),Ridge(alpha=100.))
cross=np.full(len(y),np.nan)
train=np.flatnonzero(~hold)
for fold in range(3):
 test=(episodes//4)%3==fold;fit=(~hold)&(~test);val=(~hold)&test
 v=value_model().fit(state[fit],y[fit]);cross[val]=v.predict(state[val])
assert np.isfinite(cross[~hold]).all()
v=value_model().fit(state[~hold],y[~hold]);cross[hold]=v.predict(state[hold]);adv=y-cross
started=time.perf_counter();reg=HistGradientBoostingRegressor(**plan['parameters']).fit(x[~hold],adv[~hold]);seconds=time.perf_counter()-started
artifact={**base,'estimator':reg,'state_value_model':v,'target_kind':'global_remaining_find_cost_advantage','training_plan':plan,'requires_graph_embedding':True,'minimum_gain_seconds':.1,'feature_count':x.shape[1]}
p=OUT/'candidate.joblib';joblib.dump(artifact,p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'candidate_frozen_before_validation.json').write_text(json.dumps(dict(sha256=sha,fit_seconds=seconds,heldout_labels_used_for_fitting=False),indent=2))
pred=reg.predict(x[hold]);before=float(mean_squared_error(y[hold],cross[hold])**.5);after=float(mean_squared_error(y[hold],cross[hold]+pred)**.5)
correlation=float(np.corrcoef(pred,adv[hold])[0,1])
result=dict(candidate_sha256=sha,samples=len(x),fit_seconds=seconds,state_value_rmse=before,global_action_value_rmse=after,heldout_advantage_correlation=correlation,eligible_for_short_closed_loop=after<=before*1.05 and correlation>0,new_simulations=0,goal_complete=False,acceptance_claim=False,warning='These are action-value prediction checks, not causal improvement or independent H300 acceptance')
(OUT/'offline_validation_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
