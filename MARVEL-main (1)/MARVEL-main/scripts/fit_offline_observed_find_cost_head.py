"""One CPU supervised fit on already audited episodes; no simulation or PPO training."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch,joblib,sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from paired_gain_calibration import candidate_representations
from observed_find_cost_features import find_cost_features,FEATURE_PROTOCOL
OLD=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19';OUT=ROOT/'artifacts/extended_offline_find_cost_head_20261015_attempt28_retry01';OUT.mkdir(exist_ok=False)
plan=dict(source='256 already completed attempt19 episodes; behavior-policy observed Find delay labels',train_group_rule='episode//4 modulo5 !=0',heldout_group_rule='episode//4 modulo5 ==0',feature_protocol=FEATURE_PROTOCOL,estimator='sklearn HistGradientBoostingRegressor',parameters=dict(loss='squared_error',max_iter=150,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=5.,learning_rate=.05,early_stopping=False,random_state=42),fit_calls=1,new_simulations=0,ppo_updates=0,trainable='Only offline regression head; frozen GPPO feature extractor and native weights preserved',preflight_gate='Heldout RMSE at least5% lower than both unchanged nominal travel-time and training-mean baselines; then closed-loop current-source validation required',limitations='Historical labels belong to old Safety execution and behavior policy; this is transfer supervision, not current-source acceptance',reserved_final_seeds_used=False,goal_complete=False,acceptance_claim=False)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2));torch.set_num_threads(1)
manifest=json.loads((OLD/'manifest.json').read_text())
assert manifest['source_hashes']['integrations/gppo/search_scenario.py']==hashlib.sha256((ROOT/'integrations/gppo/search_scenario.py').read_bytes()).hexdigest()
warm=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth';warm_sha=hashlib.sha256(warm.read_bytes()).hexdigest();assert warm_sha==manifest['initial_sha256']
cp=torch.load(warm,map_location='cpu',weights_only=False);model=GPPOActorCritic(edge_feature_dim=9);model.load_state_dict(cp['model']);model.eval()
x=[];y=[];groups=[];nominal=[];meta=[];excluded={};sources={}
def skip(reason):excluded[reason]=excluded.get(reason,0)+1
started=time.perf_counter()
for ep in range(256):
 folder=OLD/f'episode_{ep:03d}';records=json.loads((folder/'actor_inputs.json').read_text());decisions=json.loads((folder/'gppo/decisions.json').read_text());events=json.loads((folder/'gppo/events.json').read_text());cfg=json.loads((folder/'gppo/config.json').read_text());dt=float(cfg['scenario']['duration']['step_dt'])
 for f in ['actor_inputs.json','gppo/decisions.json','gppo/events.json']:sources[f'episode_{ep:03d}/{f}']=hashlib.sha256((folder/f).read_bytes()).hexdigest()
 found={}
 for ev in events:
  data=ev.get('data',{})
  if ev['type']=='target_found' and data.get('detection_mode')=='sensor_visible_cell' and data.get('task_id')=='T2_target_search':
   idx=int(data['target_index']);at=(int(ev['step'])+1)*dt
   if idx not in found or at<found[idx][0]:found[idx]=(at,int(data['robot_id']))
 # Source generator enumerates heat IDs and then survivors in the same order.
 # Cross-check every available abstract completion notification against exact
 # sensor times; index association is only a label-side protocol assertion.
 for ev in events:
  if ev['type']=='gppo_search_heat_serviced' and ev['data'].get('reason')=='target_found':
   hid=int(ev['data']['heat_id']);assert hid in found
   assert -1e-7<=(ev['step']+1)*dt-found[hid][0]<=1.+1e-7,(ep,hid)
 for r in records:
  tid=int(r['chosen']['subtask_id']);uid=int(r['chosen']['uav_id'])
  if not 100000<=tid<100004:skip('non_search');continue
  heat=tid-100000
  if heat not in found:skip('censored_target');continue
  at,detector=found[heat];delay=at-float(r['time'])
  if delay<=0:skip('already_found_at_decision');continue
  if detector!=uid:skip('found_by_other_uav_not_own_eta');continue
  if any(float(q['time'])>float(r['time']) and float(q['time'])<at and int(q['chosen']['subtask_id'])==tid for q in records):skip('later_reassignment_before_find');continue
  match=[d for d in decisions if d['chosen'] and float(d['current_time'])==float(r['time']) and int(d['chosen']['uav_id'])==uid and int(d['chosen']['subtask_id'])==tid]
  assert len(match)==1,(ep,r['time'],tid,uid);raw=match[0]
  assert raw['uav_ids']==r['uav_ids'] and raw['task_ids']==r['task_ids'] and raw['mask']==r['batch']['action_mask'][0]
  ti,ui=divmod(int(r['action']),len(r['uav_ids']));assert r['task_ids'][ti]==tid and r['uav_ids'][ui]==uid
  batch={k:torch.tensor(v) for k,v in r['batch'].items()};z=candidate_representations(model,batch,[r['action']])[0]
  f=find_cost_features(raw,np.asarray(r['batch']['edge_features'])[0],ti,ui,r['time'],cfg.get('dynamic_obstacles',[]),dt,z)
  x.append(f);y.append(delay);groups.append(ep);nominal.append(float(raw['raw_edge_features'][ti][ui][0]));meta.append(dict(episode=ep,seed=cfg['scenario']['random_seed'],uavs=len(r['uav_ids']),dynamic=bool(cfg.get('dynamic_obstacles')),time=r['time'],target=heat,chosen_uid=uid,actual_find=at,delay=delay))
 if ep%32==31:print('EXTRACTED',ep+1,'episodes',len(y),'strict own-Find labels',flush=True)
x=np.asarray(x);y=np.asarray(y);groups=np.asarray(groups);nominal=np.asarray(nominal);hold=(groups//4)%5==0
assert len(y)>500 and len(set(groups[hold]))>=40
coverage={}
for split,selection in [('train',~hold),('heldout',hold)]:
 counts={}
 for i in np.flatnonzero(selection):
  key=f"{meta[i]['uavs']}_{'dynamic' if meta[i]['dynamic'] else 'static'}";counts[key]=counts.get(key,0)+1
 assert len(counts)==4 and min(counts.values())>=20,counts
 coverage[split]=counts
np.savez_compressed(OUT/'observed_training_arrays.npz',features=x,delay=y,episodes=groups,nominal_travel_seconds=nominal,holdout=hold)
(OUT/'label_provenance.json').write_text(json.dumps(dict(labels=meta,excluded=excluded,source_hashes=sources,coverage=coverage,feature_count=x.shape[1],extraction_seconds=time.perf_counter()-started),indent=2))
reg=HistGradientBoostingRegressor(**plan['parameters']);fit_started=time.perf_counter();reg.fit(x[~hold],y[~hold]);fit_seconds=time.perf_counter()-fit_started
artifact=dict(estimator=reg,feature_protocol=FEATURE_PROTOCOL,feature_count=x.shape[1],frozen_encoder=str(warm),frozen_encoder_sha256=warm_sha,training_source=str(OLD),training_groups=sorted(set(map(int,groups[~hold]))),fit_calls=1,training_plan=plan)
candidate=OUT/'candidate.joblib';joblib.dump(artifact,candidate);sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
(OUT/'candidate_frozen_before_heldout.json').write_text(json.dumps(dict(sha256=sha,fit_calls=1,fit_seconds=fit_seconds,heldout_labels_used_for_fit=False,sklearn_version=sklearn.__version__,goal_complete=False),indent=2))
prediction=reg.predict(x[hold]);constant=np.full(int(hold.sum()),float(y[~hold].mean()))
def metrics(p):return dict(mae=float(mean_absolute_error(y[hold],p)),rmse=float(mean_squared_error(y[hold],p)**.5))
scores=dict(learned=metrics(prediction),nominal_travel=metrics(nominal[hold]),train_mean=metrics(constant))
eligible=scores['learned']['rmse']<.95*min(scores['nominal_travel']['rmse'],scores['train_mean']['rmse'])
result=dict(candidate_sha256=sha,samples=len(y),training_samples=int((~hold).sum()),heldout_samples=int(hold.sum()),coverage=coverage,fit_seconds=fit_seconds,heldout=scores,eligible_for_closed_loop_preflight=eligible,acceptance_claim=False,goal_complete=False,new_simulations=0,ppo_updates=0,source_transfer_warning=plan['limitations'])
(OUT/'offline_validation_complete.json').write_text(json.dumps(result,indent=2));assert hashlib.sha256(warm.read_bytes()).hexdigest()==warm_sha;print('OFFLINE_COST_VALIDATION',json.dumps(result),flush=True)
