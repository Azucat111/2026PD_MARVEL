"""Read-only audit: did PPO change probabilities in its recorded advantage direction?"""
from pathlib import Path
import sys,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from integrations.gppo.gppo_model import GPPOActorCritic
D=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19'
OUT=ROOT/'artifacts/extended_checkpoint_common_comparison_20261015_attempt33'
torch.set_num_threads(1)
history=json.loads((D/'progress.json').read_text());updates=json.loads((D/'updates.json').read_text());args=json.loads((D/'manifest.json').read_text())['args']
batches=[];allrows=[];between_ss=0.;total_ss=0.;max_kl_error=0.
for bi,up in enumerate(updates):
 cp_path=D/f'checkpoint_batch_{bi:03d}.pth';cp=torch.load(cp_path,map_location='cpu',weights_only=False)
 with torch.random.fork_rng(devices=[]):m=GPPOActorCritic(edge_feature_dim=9)
 m.load_state_dict(cp['model']);m.eval()
 records=[]
 for h in history[bi*8:(bi+1)*8]:
  rr=json.loads((D/f"episode_{h['episode']:03d}"/'actor_inputs.json').read_text())
  assert len(rr)==len(h['returns'])
  for r,y in zip(rr,h['returns']):records.append(dict(record=r,episode=h['episode'],role=1 if 100000<=r['chosen']['subtask_id']<200000 else 2,target=float(y)))
 raw=np.asarray([r['target']-r['record']['value'] for r in records],dtype=np.float32);roles=np.asarray([r['role'] for r in records]);adv=np.zeros_like(raw)
 for role in set(roles.tolist()):
  idx=roles==role;x=raw[idx];adv[idx]=(x-x.mean())/x.std() if len(x)>1 and x.std()>1e-8 else 0.
 ratios=[];logratios=[]
 for r,a in zip(records,adv):
  z=r['record'];batch={k:torch.tensor(v) for k,v in z['batch'].items()}
  with torch.no_grad():o=m(**batch);dist=torch.distributions.Categorical(logits=o['flat_logits'][0]/args['temperature']);newlog=float(dist.log_prob(torch.tensor(z['action'])))
  delta=newlog-z['log_prob'];ratios.append(np.exp(delta));logratios.append(delta)
  allrows.append(dict(batch=bi,episode=r['episode'],role=r['role'],time=z['time'],chosen=z['chosen'],advantage=float(a),log_probability_change=delta,same_sign=bool(a*delta>0),return_=r['target']))
 logratios=np.asarray(logratios);ratio=np.asarray(ratios);kl=float(np.mean(ratio-1-logratios))
 expected=up['epochs'][-1]['approx_kl'];max_kl_error=max(max_kl_error,abs(kl-expected))
 inds=[i for i,r in enumerate(records) if r['role']==1];vals=adv[inds];ep=np.asarray([records[i]['episode'] for i in inds]);mean=vals.mean();total_ss+=float(np.square(vals-mean).sum())
 for e in np.unique(ep):
  v=vals[ep==e];between_ss+=len(v)*float((v.mean()-mean)**2)
 batches.append(dict(batch=bi,records=len(records),approx_kl_recomputed=kl,approx_kl_recorded=expected,mean_advantage_times_probability_change=float(np.mean(adv*logratios))))
stats={}
for role,name in [(1,'Search'),(2,'Relay')]:
 rows=[r for r in allrows if r['role']==role and abs(r['advantage'])>1e-8]
 stats[name]=dict(records=len(rows),fraction_probability_change_in_advantage_direction=sum(r['same_sign'] for r in rows)/len(rows),mean_advantage_times_log_probability_change=float(np.mean([r['advantage']*r['log_probability_change'] for r in rows])))
result=dict(training='attempt19 completed256',batches=len(batches),records=len(allrows),max_completed_update_kl_recompute_error=max_kl_error,role_statistics=stats,search_advantage_variance_between_episodes_fraction=between_ss/total_ss,batch_objective_direction_positive=sum(r['mean_advantage_times_probability_change']>0 for r in batches),batches_detail=batches,records_detail=allrows,new_optimizer_updates=0,new_physical_simulations=0,causal_claim=False,interpretation_limit='Correct probability updates toward recorded advantage do not prove those labels identify causally better assignments. Between-episode variation also contains physical difficulty, decision timing and critic error.')
(OUT/'training19_update_direction_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['batches_detail','records_detail']}),flush=True)
