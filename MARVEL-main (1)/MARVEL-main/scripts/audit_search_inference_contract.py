"""Saved-input inference audit only; no rollout, optimizer or production changes."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.checkpoint import load_frozen_gppo
torch.set_num_threads(1)
checkpoint=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
model,_=load_frozen_gppo(checkpoint,device='cpu')
parameters_before={k:v.detach().clone() for k,v in model.state_dict().items()}
rows=[]
for relative in sys.argv[1:]:
    file=ROOT/relative/'gppo/decisions.json'
    records=json.loads(file.read_text())
    for index,r in enumerate(records):
        if not r['chosen']:continue
        arrays={'uav_features':np.asarray(r['model_uav_features']),
            'task_features':np.asarray(r['model_task_features']),
            'edge_features':np.asarray(r['model_edge_features']),
            'capability_matrix':np.asarray(r['capability_matrix']),
            'precedence_matrix':np.asarray(r['precedence_matrix']),
            'action_mask':np.asarray(r['mask'])}
        T,U=arrays['action_mask'].shape
        def infer(a):
            batch={k:torch.as_tensor(v,dtype=torch.bool if k in ('capability_matrix','precedence_matrix','action_mask') else torch.float32).unsqueeze(0) for k,v in a.items()}
            batch['uav_valid_mask']=torch.ones((1,U),dtype=torch.bool)
            batch['task_valid_mask']=torch.ones((1,T),dtype=torch.bool)
            with torch.no_grad():return model(**batch)['pair_logits'][0].numpy()
        original=infer(arrays);ti,ui=np.unravel_index(original.argmax(),original.shape)
        assert {'subtask_id':r['task_ids'][ti],'uav_id':r['uav_ids'][ui]}==r['chosen']
        perm=np.arange(U)[::-1];a={k:v.copy() for k,v in arrays.items()}
        a['uav_features']=a['uav_features'][perm]
        for key in ('edge_features','capability_matrix','action_mask'):a[key]=a[key][:,perm]
        restored=infer(a)[:,np.argsort(perm)]
        valid=~arrays['action_mask'].astype(bool)
        error=float(np.max(np.abs(original[valid]-restored[valid])))
        assert error<1e-5,(relative,index,error)
        legal=np.flatnonzero(valid[ti]);cost=np.asarray(r['observed_total_time'])[ti]
        best=int(legal[np.argmin(cost[legal])])
        derivatives={}
        for label,col in [('chosen',ui),('fastest',best)]:
            # Conditional feature sensitivity, not a physically realizable state.
            plus={k:v.copy() for k,v in arrays.items()};minus={k:v.copy() for k,v in arrays.items()}
            eps=1e-3
            for ch in (0,2):
                plus['edge_features'][ti,col,ch]+=eps
                minus['edge_features'][ti,col,ch]-=eps
            derivatives[label]=float((infer(plus)[ti,col]-infer(minus)[ti,col])/(2*eps))
        rows.append({'case':relative,'decision':index,'step':r['step'],
            'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'exact_replay':True,'permutation_max_logit_error':error,
            'chosen':r['chosen'],'chosen_observed_time':float(cost[ui]),
            'fastest_uav':r['uav_ids'][best],'fastest_observed_time':float(cost[best]),
            'conditional_score_derivative_for_increasing_normalized_travel':derivatives})
assert all(torch.equal(parameters_before[k],v) for k,v in model.state_dict().items())
out=ROOT/'artifacts/search_inference_contract_20261010.json'
assert not out.exists()
out.write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'rollouts':0,'training':False,'model_parameters_unchanged':True,
    'production_changed':False,'results':rows,
    'limitation':'Permutation/replay establish input-index execution correctness only. Conditional derivatives do not prove achievable efficiency or global impossibility.'},indent=2))
print(json.dumps({'decisions':len(rows),'all_exact_replay':True,'max_permutation_error':max(r['permutation_max_logit_error'] for r in rows),'parameters_unchanged':True,'conditional_derivatives':[r['conditional_score_derivative_for_increasing_normalized_travel'] for r in rows]}))
