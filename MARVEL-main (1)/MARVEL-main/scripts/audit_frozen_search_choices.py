"""Exact offline replay of saved frozen neural decisions; no rollout/training."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)
from integrations.gppo.checkpoint import load_frozen_gppo
checkpoint=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
model,_=load_frozen_gppo(checkpoint,device='cpu')
for relative in sys.argv[1:]:
    folder=ROOT/relative
    source=folder/'gppo/decisions.json'
    records=json.loads(source.read_text());results=[]
    for index,r in enumerate(records):
        mask=np.asarray(r['mask'],dtype=bool)
        if not r['chosen']:continue
        arrays={'uav_features':r['model_uav_features'],'task_features':r['model_task_features'],
            'edge_features':r['model_edge_features'],'capability_matrix':r['capability_matrix'],
            'precedence_matrix':r['precedence_matrix'],'action_mask':r['mask']}
        batch={k:torch.as_tensor(v,dtype=torch.bool if k in ('capability_matrix','precedence_matrix','action_mask') else torch.float32).unsqueeze(0) for k,v in arrays.items()}
        batch['uav_valid_mask']=torch.ones((1,len(r['uav_ids'])),dtype=torch.bool)
        batch['task_valid_mask']=torch.ones((1,len(r['task_ids'])),dtype=torch.bool)
        with torch.no_grad():out=model(**batch)
        flat=int(out['flat_logits'][0].argmax());ti,ui=divmod(flat,len(r['uav_ids']))
        selected={'subtask_id':r['task_ids'][ti],'uav_id':r['uav_ids'][ui]}
        assert selected==r['chosen'],(relative,index,selected,r['chosen'])
        costs=np.asarray(r['observed_total_time'])[ti]
        legal=np.flatnonzero(~mask[ti]);best=int(legal[np.argmin(costs[legal])])
        results.append({'decision_index':index,'step':r['step'],'exact_neural_replay':True,
            'selected':selected,'search_task':bool(r['model_task_features'][ti][7]),
            'chosen_observed_seconds':float(costs[ui]),'minimum_same_task_seconds':float(costs[best]),
            'minimum_same_task_uav':r['uav_ids'][best],'legal_same_task_count':len(legal),
            'chosen_over_minimum':float(costs[ui]/costs[best]) if costs[best]>0 else None})
    summary={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'rollouts':0,'training':False,'production_decisions_changed':False,'results':results,
        'limitation':'Observed costs are estimates; ranking mismatch does not prove global policy impossibility.'}
    (folder/'frozen_choice_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'case':relative,'exact_replays':len(results),'search_choices':[v for v in results if v['search_task']] }),flush=True)
