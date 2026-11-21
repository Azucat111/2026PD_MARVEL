"""Declared ckpt50 shadow vs final80 on fixed saved inputs only; no screening other checkpoints."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.checkpoint import load_frozen_gppo
torch.set_num_threads(1)
base=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen')
paths={role:base/file for role,file in [('final80','gppo_phase14_v9_ckpt80.pth'),('declared_shadow50','gppo_phase14_v9_ckpt50_shadow.pth')]}
models={role:load_frozen_gppo(p,device='cpu')[0] for role,p in paths.items()}
rows=[]
for relative in sys.argv[1:]:
    source=ROOT/relative/'gppo/decisions.json'
    for index,r in enumerate(json.loads(source.read_text())):
        if not r['chosen']:continue
        original_ti=r['task_ids'].index(r['chosen']['subtask_id'])
        if r['model_task_features'][original_ti][7]<.5:continue
        arrays={'uav_features':r['model_uav_features'],'task_features':r['model_task_features'],
            'edge_features':r['model_edge_features'],'capability_matrix':r['capability_matrix'],
            'precedence_matrix':r['precedence_matrix'],'action_mask':r['mask']}
        batch={k:torch.as_tensor(v,dtype=torch.bool if k in ('capability_matrix','precedence_matrix','action_mask') else torch.float32).unsqueeze(0) for k,v in arrays.items()}
        U=len(r['uav_ids']);T=len(r['task_ids'])
        batch['uav_valid_mask']=torch.ones((1,U),dtype=torch.bool);batch['task_valid_mask']=torch.ones((1,T),dtype=torch.bool)
        choices={}
        for role,model in models.items():
            with torch.no_grad():out=model(**batch)
            ti,ui=divmod(int(out['flat_logits'][0].argmax()),U)
            assert not r['mask'][ti][ui]
            cost=np.asarray(r['observed_total_time'])[ti];legal=np.flatnonzero(~np.asarray(r['mask'][ti],dtype=bool))
            best=int(legal[np.argmin(cost[legal])])
            choices[role]={'task_id':r['task_ids'][ti],'uav_id':r['uav_ids'][ui],
                'observed_cost':float(cost[ui]),'same_task_rule_minimum':float(cost[best]),
                'same_task_cost_ratio':float(cost[ui]/cost[best])}
        assert choices['final80']['task_id']==r['chosen']['subtask_id'] and choices['final80']['uav_id']==r['chosen']['uav_id']
        rows.append({'case':relative,'decision':index,'step':r['step'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'choices':choices})
summary={role:{'sum_observed_cost':sum(r['choices'][role]['observed_cost'] for r in rows),
    'sum_same_task_rule_minimum':sum(r['choices'][role]['same_task_rule_minimum'] for r in rows),
    'mean_same_task_cost_ratio':float(np.mean([r['choices'][role]['same_task_cost_ratio'] for r in rows]))} for role in models}
result={'models':{role:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for role,p in paths.items()},
    'rollouts':0,'training':False,'production_changed':False,'other_checkpoints_screened':False,
    'exact_final80_replay':True,'summary':summary,'rows':rows,
    'limitation':'Saved-input shadow counterfactual, not a shadow-policy trajectory or >=15% detection improvement proof; changing task choice changes later assignment state.'}
out=ROOT/'artifacts/search_declared_shadow_comparison_20261010.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps({'decisions':len(rows),'summary':summary,'rollouts':0,'training':False,'production_changed':False}))
