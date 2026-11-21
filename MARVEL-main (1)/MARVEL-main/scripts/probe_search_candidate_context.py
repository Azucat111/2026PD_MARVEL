"""Conditional saved-feature probe, not deployed policy or physical evaluation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.task_graph import TaskGraph
torch.set_num_threads(1)
p=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
model,_=load_frozen_gppo(p,device='cpu');rows=[]
def infer(u,t,e,cap,pre,mask):
    data={k:torch.as_tensor(v,dtype=torch.bool if k in ('capability_matrix','precedence_matrix','action_mask') else torch.float32).unsqueeze(0) for k,v in {'uav_features':u,'task_features':t,'edge_features':e,'capability_matrix':cap,'precedence_matrix':pre,'action_mask':mask}.items()}
    data['uav_valid_mask']=torch.ones((1,len(u)),dtype=torch.bool);data['task_valid_mask']=torch.ones((1,len(t)),dtype=torch.bool)
    with torch.no_grad():return model(**data)['pair_logits'][0].numpy()
for relative in sys.argv[1:]:
    source=ROOT/relative/'gppo/decisions.json';records=json.loads(source.read_text());seen=set()
    for r in records:
        if not r['chosen'] or r['step'] in seen:continue
        seen.add(r['step']);mask=np.asarray(r['mask'],dtype=bool);cost=np.asarray(r['observed_total_time'])
        rawt=np.asarray(r['raw_task_features']);active=np.flatnonzero((np.asarray(r['model_task_features'])[:,7]>.5)&(~mask).any(axis=1))
        for ti in active:
            legal=np.flatnonzero(~mask[ti]);pool=sorted(legal,key=lambda u:(cost[ti,u],r['uav_ids'][u]))[:4]
            if len(pool)<4:continue
            scoped=np.ones_like(mask);scoped[ti,pool]=False
            full=infer(r['model_uav_features'],r['model_task_features'],r['model_edge_features'],r['capability_matrix'],r['precedence_matrix'],scoped)
            full_ui=int(full[ti].argmax())
            graph=TaskGraph.__new__(TaskGraph);graph.uav_states=[None]*4;graph.subtasks=[None]*len(rawt)
            graph.current_time=r['current_time'];graph.uav_features=np.asarray(r['raw_uav_features'])[pool]
            graph.task_features=rawt;graph.edge_features=np.asarray(r['raw_edge_features'])[:,pool]
            u,t,e=graph._build_model_features()
            small=infer(u,t,e,np.asarray(r['capability_matrix'])[:,pool],r['precedence_matrix'],scoped[:,pool])
            small_ui=pool[int(small[ti].argmax())]
            rows.append({'case':relative,'step':r['step'],'task_id':r['task_ids'][ti],
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'candidate_ids':[r['uav_ids'][i] for i in pool],
                'same_pool_full_context_uav':r['uav_ids'][full_ui],
                'four_uav_context_uav':r['uav_ids'][small_ui],
                'full_context_observed_cost':float(cost[ti,full_ui]),
                'four_context_observed_cost':float(cost[ti,small_ui]),
                'rule_pool_minimum_cost':float(cost[ti,pool[0]])})
f=sum(r['full_context_observed_cost'] for r in rows);s=sum(r['four_context_observed_cost'] for r in rows);b=sum(r['rule_pool_minimum_cost'] for r in rows)
result={'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rollouts':0,'training':False,'production_changed':False,
    'declared_scope':'K=4 public observed-cost pool, same pool in both contexts; same raw features, normalization recomputed by frozen implementation; only selected Search row active; this changes the decision problem and is NOT an exact replay.',
    'limitation':'Conditional input-domain probe only. No legal path/true detection/performance proof. Do not attribute candidate-budget gains to neural adaptation.',
    'full_context_cost_sum':f,'four_context_cost_sum':s,'rule_minimum_sum':b,
    'changed_choices':sum(r['same_pool_full_context_uav']!=r['four_uav_context_uav'] for r in rows),'rows':rows}
out=ROOT/'artifacts/search_candidate_context_probe_20261010.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
