"""Observed saved-graph oracle for assignment cost only; not detection evidence."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for relative in sys.argv[1:]:
    source=ROOT/relative/'gppo/decisions.json'
    records=json.loads(source.read_text());steps=sorted({r['step'] for r in records if r['chosen']})
    for step in steps:
        group=[r for r in records if r['step']==step and r['chosen']]
        r=group[0];mask=np.asarray(r['mask'],dtype=bool);cost=np.asarray(r['observed_total_time'],dtype=float)
        task=np.asarray(r['raw_task_features']);search=np.asarray(r['model_task_features'])[:,7]>0.5
        active=np.flatnonzero(search & (~mask).any(axis=1))
        if not len(active):continue
        priorities=task[active,0]
        if not np.all(priorities==priorities[0]):raise ValueError('Different priority tiers need separate lexicographic audit')
        localcost=cost[active];localmask=mask[active]
        # Dummy assignments dominate every possible cost sum, maximizing cardinality.
        penalty=(float(np.max(np.abs(localcost)))+1.)*(len(active)+1)*100.
        matrix=np.concatenate([np.where(localmask,penalty*100.,localcost),np.full((len(active),len(active)),penalty)],axis=1)
        ai,ui=linear_sum_assignment(matrix)
        optimal=[(int(active[a]),int(u)) for a,u in zip(ai,ui) if u<cost.shape[1] and not localmask[a,u]]
        used=set();greedy=[]
        for ti in sorted(active,key=lambda i:r['task_ids'][i]):
            candidates=[u for u in range(cost.shape[1]) if not mask[ti,u] and u not in used]
            if not candidates:continue
            u=min(candidates,key=lambda u:(cost[ti,u],r['uav_ids'][u]));used.add(u);greedy.append((int(ti),u))
        neural=[];initial_valid=True
        for item in group:
            chosen=item['chosen']
            if chosen['subtask_id'] not in r['task_ids']:continue
            ti=r['task_ids'].index(chosen['subtask_id']);u=r['uav_ids'].index(chosen['uav_id'])
            if ti not in active:continue
            neural.append((ti,u));initial_valid &= not bool(mask[ti,u])
        def value(pairs):return float(sum(cost[t,u] for t,u in pairs))
        def ids(pairs):return [{'task_id':r['task_ids'][t],'uav_id':r['uav_ids'][u],'observed_time':float(cost[t,u])} for t,u in pairs]
        matched=len(optimal)==len(greedy)==len(neural)
        savings=1.-value(optimal)/value(greedy) if matched and value(greedy)>0 else None
        rows.append({'case':relative,'step':step,'same_priority_search_tasks':len(active),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'optimal':ids(optimal),'same_graph_priority_greedy':ids(greedy),'saved_neural':ids(neural),
            'optimal_sum':value(optimal),'greedy_sum':value(greedy),'neural_initial_graph_sum':value(neural),
            'cardinality_matched':matched,'neural_initial_mask_valid':initial_valid,
            'joint_cost_savings_fraction_over_greedy':savings})
comparable=[r for r in rows if r['cardinality_matched']]
greedy=sum(r['greedy_sum'] for r in comparable);opt=sum(r['optimal_sum'] for r in comparable)
out=ROOT/'artifacts/search_joint_allocation_audit_20261010.json'
assert not out.exists()
result={'rollouts':0,'training':False,'production_changed':False,
    'oracle_inputs':'saved public observed cost matrices and valid masks only; no true obstacle or hidden target',
    'metric':'snapshot sum of estimated travel plus processing, not true detection latency',
    'limitation':'Optimal assignment does not prove a legal complete motion path, deployed recovery, or >=15% task-efficiency gain. Sequential route-support gates may differ; marked initial-invalid neural pairs are not strict regret comparisons.',
    'greedy_total':greedy,'optimal_total':opt,'estimated_joint_savings_fraction':1.-opt/greedy if greedy>0 else None,
    'rows':rows}
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
print(json.dumps([{'case':r['case'],'step':r['step'],'tasks':r['same_priority_search_tasks'],'savings':r['joint_cost_savings_fraction_over_greedy'],'mask_valid':r['neural_initial_mask_valid']} for r in rows]))
