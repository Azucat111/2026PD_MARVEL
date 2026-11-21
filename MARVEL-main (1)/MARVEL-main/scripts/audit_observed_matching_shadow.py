"""Cost-only shadow of matching on previously captured public actor graph tensors."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
folders=['extended_joint_batch_30_static_9157_20261010_attempt1','extended_joint_batch_60_dynamic_9157_20261010_attempt1','extended_joint_batch_30_dynamic_9157_20261010_attempt1','extended_joint_batch_60_static_9157_20261010_attempt1']
rows=[]
for folder in folders:
    data=json.loads((ROOT/'artifacts'/folder/'gppo/actor_inputs.json').read_text())
    for item in data:
        if len(item['task_ids'])<2 or not all(100000<=t<200000 for t in item['task_ids']):continue
        cost=np.asarray(item['batch']['edge_features'][0])[:,:,2];legal=~np.asarray(item['batch']['action_mask'][0],dtype=bool)
        penalty=(float(cost.max())+1)*(len(cost)+1)
        matrix=np.full((len(cost),cost.shape[1]+len(cost)),penalty);matrix[:,:cost.shape[1]]=np.where(legal,cost,4*penalty)
        ts,us=linear_sum_assignment(matrix);pairs=[(float(cost[t,u]),item['task_ids'][t],item['uav_ids'][u],int(t),int(u)) for t,u in zip(ts,us) if u<cost.shape[1] and legal[t,u]]
        if not pairs:continue
        *_,t,u=min(pairs)
        remaining=legal.copy();greedy=[]
        while remaining.any():
            choices=[(float(cost[ti,ui]),item['task_ids'][ti],item['uav_ids'][ui],int(ti),int(ui)) for ti,ui in np.argwhere(remaining)]
            row=min(choices);greedy.append(row);remaining[row[3],:]=False;remaining[:,row[4]]=False
        priorities=item['batch']['task_features'][0]
        same_priority=len({row[0] for row in priorities})==1
        rows.append(dict(folder=folder,time=item['time'],old_actor_action=item['action'],matching_action=t*cost.shape[1]+u,actor_order_changed=t*cost.shape[1]+u!=item['action'],greedy_first_action=greedy[0][3]*cost.shape[1]+greedy[0][4],greedy_pairs=[[r[1],r[2]] for r in greedy],matching_pairs=[[r[1],r[2]] for r in pairs],greedy_cost=sum(r[0] for r in greedy),matching_cost=sum(r[0] for r in pairs),same_priority=same_priority,public_cost_only=True))
result=dict(rows=rows,lower_matching_cost_states=sum(r['matching_cost']<r['greedy_cost']-1e-7 for r in rows),states=len(rows),acceptance_claim=False)
with (ROOT/'artifacts/extended_matching_shadow_corrected_9157_20261010.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
