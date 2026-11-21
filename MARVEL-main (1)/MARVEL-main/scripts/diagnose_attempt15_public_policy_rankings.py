"""Read-only policy ranking diagnosis on saved public training graphs; not acceptance."""
from pathlib import Path
import sys,json,math
from collections import Counter
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gppo_parallel_rollouts import load_captured_episode
from integrations.gppo.gppo_model import GPPOActorCritic
folder=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15'
cp=torch.load(folder/'checkpoint_batch_007.pth',map_location='cpu',weights_only=False);model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model']);model.eval();torch.set_num_threads(1)
counts=Counter();search=[];examples=[]
for i in range(64):
 row,tr=load_captured_episode(folder,i,'cpu');decisions=json.loads((folder/f'episode_{i:03d}/gppo/decisions.json').read_text())
 for t in tr:
  record=t['record'];matches=[d for d in decisions if abs(d['current_time']-record['time'])<1e-9 and d['chosen']==record['chosen'] and d['uav_ids']==record['uav_ids'] and d['task_ids']==record['task_ids']];assert len(matches)==1;d=matches[0]
  with torch.no_grad():o=model(**t['batch'])
  pred=int(o['flat_logits'][0].argmax());teacher=t['teacher'];n=len(record['uav_ids']);pt,pu=divmod(pred,n);tt,tu=divmod(teacher,n)
  pr=int(np.argmax(d['raw_task_features'][pt][6:10]));rr=int(np.argmax(d['raw_task_features'][tt][6:10]));counts['records']+=1
  label='same_action' if pred==teacher else ('different_role' if pr!=rr else ('different_task' if pt!=tt else 'same_task_different_uav'));counts[label]+=1
  if rr==1 and pr==1:
   edges=t['batch']['edge_features'][0];rowdiag=dict(episode=i,time=record['time'],category=label,teacher_task=record['task_ids'][tt],teacher_uav=record['uav_ids'][tu],pred_task=record['task_ids'][pt],pred_uav=record['uav_ids'][pu],raw_time_delta=float(d['raw_edge_features'][pt][pu][2]-d['raw_edge_features'][tt][tu][2]),teacher_heading_cos=float(edges[tt,tu,3]),pred_heading_cos=float(edges[pt,pu,3]));search.append(rowdiag)
   if pred!=teacher and len(examples)<20:examples.append(rowdiag)
result=dict(counts=dict(counts),search_records=len(search),mean_raw_time_delta=sum(x['raw_time_delta'] for x in search)/max(1,len(search)),examples=examples,scope='Final checkpoint re-ranked saved public training inputs only. These are training data and do not establish independent efficiency gain; no counterfactual physical execution claimed.',acceptance_claim=False)
with (folder/'final_policy_public_ranking_diagnosis.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
