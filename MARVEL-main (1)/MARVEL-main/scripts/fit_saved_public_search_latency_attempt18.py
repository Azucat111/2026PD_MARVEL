"""Offline public-input Find latency calibration diagnostic; no policy/runtime edits."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18'
torch.set_num_threads(1);torch.manual_seed(63076);torch.use_deterministic_algorithms(True)
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt18.json').read_text());hashes={**plan['source_hashes'],**plan['pipeline_hashes']}
def guard():return all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert guard()
f=folder/'readonly_search_latency_public_dataset.json';data=json.loads(f.read_text());rows=data['rows'];assert not data['label_problems']
assert not ({r['seed'] for r in rows}&{9169,101101,202202,303303})
X=np.asarray([r['public_features'] for r in rows],dtype=np.float64);y=np.asarray([r['actual_remaining_find_seconds'] for r in rows],dtype=np.float64)
assert X.shape==(982,len(data['feature_names'])) and np.isfinite(X).all() and np.isfinite(y).all() and (y>=0).all()
split={k:np.asarray([i for i,r in enumerate(rows) if r['seed']%5 in mods],dtype=int) for k,mods in [('train',[0,1,2]),('validation',[3]),('test',[4])]}
seed_sets={k:{rows[i]['seed'] for i in idx} for k,idx in split.items()}
assert all(not(seed_sets[a]&seed_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
mu=X[split['train']].mean(axis=0);sd=np.maximum(X[split['train']].std(axis=0),1e-6);Z=(X-mu)/sd;target=np.log1p(y)
# Fixed model family and validation-only selection. Reserved natural cases never enter.
A=np.column_stack([Z,np.ones(len(Z))]);lam=np.eye(A.shape[1]);lam[-1,-1]=0.;coef=np.linalg.solve(A[split['train']].T@A[split['train']]+lam,A[split['train']].T@target[split['train']])
ridge=np.maximum(0,np.minimum(150,np.expm1(np.clip(A@coef,-20,np.log1p(150)))))
z=torch.as_tensor(Z,dtype=torch.float32);yt=torch.as_tensor(target,dtype=torch.float32);net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],32),torch.nn.SiLU(),torch.nn.Linear(32,16),torch.nn.SiLU(),torch.nn.Linear(16,1));opt=torch.optim.Adam(net.parameters(),lr=.003)
best=None;best_val=float('inf');best_epoch=None
for epoch in range(600):
 opt.zero_grad(set_to_none=True);pred=net(z[split['train']]).flatten();loss=torch.nn.functional.smooth_l1_loss(pred,yt[split['train']],beta=.25);assert torch.isfinite(loss);loss.backward();opt.step()
 with torch.no_grad():p=torch.expm1(net(z[split['validation']]).flatten().clamp(0,float(np.log1p(150)))).numpy();score=float(np.mean(np.abs(p-y[split['validation']])))
 if score<best_val:best_val=score;best=copy.deepcopy(net.state_dict());best_epoch=epoch
net.load_state_dict(best);net.eval()
with torch.no_grad():mlp=torch.expm1(net(z).flatten().clamp(0,float(np.log1p(150)))).numpy()
constant=np.full(len(y),float(y[split['train']].mean()));point=np.asarray([r['observed_point_total_seconds'] for r in rows]);sensor_proxy=np.maximum(0,np.asarray([r['public_distance_seconds'] for r in rows])-10.)
predictions={'training_mean':constant,'observed_point_total':point,'public_distance_minus_sensor_range':sensor_proxy,'ridge_public':ridge,'mlp_public':mlp}
metrics={name:{part:dict(records=len(idx),mae=float(np.mean(np.abs(p[idx]-y[idx]))),rmse=float(np.sqrt(np.mean((p[idx]-y[idx])**2)))) for part,idx in split.items()} for name,p in predictions.items()}
selected=min(('ridge_public','mlp_public'),key=lambda n:metrics[n]['validation']['mae'])
torch.save(dict(protocol='offline-public-find-latency-diagnostic-v1',feature_names=data['feature_names'],model_state=net.state_dict(),mu=mu.tolist(),sd=sd.tolist(),ridge_coefficients=coef.tolist(),selection=selected,selected_by='validation only',dataset_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),best_epoch=best_epoch,production_deployment=False),folder/'offline_public_find_latency_prototype.pth')
assert guard()
result=dict(records=len(rows),feature_names=data['feature_names'],split_rule='seed modulo5: train0,1,2;validation3;test4',split_episode_counts={k:len(s) for k,s in seed_sets.items()},metrics=metrics,selected_by_validation=selected,best_mlp_epoch=best_epoch,public_inputs_only=True,protected_source_stable=True,unique_guarded_files=len(hashes),reserved_and_dev_seeds_unused=True,acceptance_claim=False,limitations='Factual selected Search actions only; no unselected-action labels, no policy selection or natural runtime gain proof. This is not the project15% metric.')
with (folder/'offline_public_find_latency_calibration.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
