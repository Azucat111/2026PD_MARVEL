"""Offline replay of captured observed GPPO inputs; no rollout/training.
Scale/available-row probes are diagnostic, never production decisions.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)
from integrations.gppo.checkpoint import load_frozen_gppo
CHECKPOINT=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
model,checkpoint=load_frozen_gppo(CHECKPOINT,device='cpu')
folder=ROOT/'artifacts/fair_search_input_capture_20261010'
records=json.loads((folder/'gppo/decisions.json').read_text())
results=[]
for index,record in enumerate(records):
    mask=np.asarray(record['mask'],dtype=bool)
    costs=np.asarray(record['observed_total_time'])
    full_uids=list(range(len(record['uav_ids'])))
    available=np.flatnonzero((~mask).any(axis=0)).tolist()
    cheapest=set()
    for row in range(len(record['task_ids'])):
        valid=np.flatnonzero(~mask[row])
        cheapest.update(sorted(valid,key=lambda uid:(costs[row,uid],record['uav_ids'][uid]))[:4])
    variants={'full_observed_graph':full_uids,'available_rows_only':available,
        'four_cheapest_per_task_diagnostic':sorted(cheapest)}
    def infer(columns):
        if not columns:return None
        arrays={'uav_features':np.asarray(record['model_uav_features'])[columns],
            'task_features':np.asarray(record['model_task_features']),
            'edge_features':np.asarray(record['model_edge_features'])[:,columns],
            'capability_matrix':np.asarray(record['capability_matrix'])[:,columns],
            'precedence_matrix':np.asarray(record['precedence_matrix']),
            'action_mask':mask[:,columns]}
        batch={k:torch.as_tensor(v,dtype=torch.bool if k in ('capability_matrix','precedence_matrix','action_mask') else torch.float32).unsqueeze(0) for k,v in arrays.items()}
        batch['uav_valid_mask']=torch.ones((1,len(columns)),dtype=torch.bool)
        batch['task_valid_mask']=torch.ones((1,len(record['task_ids'])),dtype=torch.bool)
        with torch.no_grad():out=model(**batch)
        flat=int(out['flat_logits'][0].argmax())
        ti,local_ui=divmod(flat,len(columns));ui=columns[local_ui]
        logits=out['flat_logits'][0].numpy().reshape(len(record['task_ids']),len(columns))
        ranking=sorted(((float(logits[t,j]),t,u) for t in range(len(record['task_ids'])) for j,u in enumerate(columns) if not mask[t,u]),reverse=True)
        return {'chosen':{'subtask_id':record['task_ids'][ti],'uav_id':record['uav_ids'][ui]},
            'chosen_observed_cost':float(costs[ti,ui]),
            'minimum_valid_observed_cost':float(costs[~mask].min()) if (~mask).any() else None,
            'top_ranked':[{'task_id':record['task_ids'][t],'uav_id':record['uav_ids'][u],'logit':v,'cost':float(costs[t,u])} for v,t,u in ranking[:5]]}
    probes={name:infer(columns) for name,columns in variants.items()}
    assert probes['full_observed_graph']['chosen']==record['chosen']
    results.append({'decision_index':index,'step':record['step'],'exact_replay':True,'variants':probes})
summary={'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    'source_decisions_sha256':hashlib.sha256((folder/'gppo/decisions.json').read_bytes()).hexdigest(),
    'production_modified':False,'rollouts':0,'results':results,
    'interpretation':'Subset encoders change policy context. These are probes, not performance or repair evidence.'}
(folder/'offline_input_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
