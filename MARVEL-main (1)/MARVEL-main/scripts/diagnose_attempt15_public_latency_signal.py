"""Saved training-only supervised signal diagnosis; never a policy acceptance claim."""
from pathlib import Path
import json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt15'
rows=[]
for i in range(64):
 folder=F/f'episode_{i:03d}'
 summary=json.loads((folder/'gppo/summary.json').read_text())
 inputs=json.loads((folder/'actor_inputs.json').read_text())
 decisions=json.loads((folder/'gppo/decisions.json').read_text())
 found={e['data']['target_index']:e['data']['robot_id'] for e in json.loads((folder/'gppo/events.json').read_text()) if e['type']=='target_found'}
 for a in inputs:
  tid=int(a['chosen']['subtask_id']);uid=int(a['chosen']['uav_id'])
  if not 100000<=tid<100004:continue
  matches=[d for d in decisions if d['chosen']==a['chosen'] and abs(d['current_time']-a['time'])<1e-8 and d['uav_ids']==a['uav_ids'] and d['task_ids']==a['task_ids']]
  assert len(matches)==1
  d=matches[0];t=d['task_ids'].index(tid);u=d['uav_ids'].index(uid)
  uf=d['raw_uav_features'][u];tf=d['raw_task_features'][t]
  distance=math.hypot(tf[10]-uf[9],tf[11]-uf[10])
  edge=a['batch']['edge_features'][0][t][u]
  rawtime=d['raw_edge_features'][t][u][2]
  heat=tid-100000;latency=max(0.,summary['metric']['capped_detection_seconds'][heat]-a['time'])
  rows.append(dict(episode=i,uid=uid,heat=heat,time=a['time'],remaining=latency,censored=heat not in found,detected_by_assigned=found.get(heat)==uid,
    features=[rawtime,distance,max(0.,distance-10.),edge[3],edge[4],float(edge[5]),a['time']],
    label_scope='Observed episode outcome only, no unexecuted action labels'))
X=np.asarray([r['features'] for r in rows]);y=np.asarray([r['remaining'] for r in rows]);groups=np.asarray([r['episode'] for r in rows])
variants={'raw_path_cost':[0],'public_range_heading':[1,2,3,4,5,6],'combined_public':[0,1,2,3,4,5,6]}
metrics={}
for name,cols in variants.items():
 pred=np.zeros(len(y))
 for fold in range(8):
  test=groups%8==fold;train=~test;xx=X[:,cols];mean=xx[train].mean(0);std=xx[train].std(0);std[std<1e-8]=1.
  zz=np.column_stack([np.ones(len(y)),(xx-mean)/std]);reg=np.eye(zz.shape[1])*.1;reg[0,0]=0.
  w=np.linalg.solve(zz[train].T@zz[train]+reg,zz[train].T@y[train]);pred[test]=np.maximum(0,zz[test]@w)
 metrics[name]=dict(episode_disjoint_folds=8,mae_seconds=float(np.abs(pred-y).mean()),rmse_seconds=float(np.sqrt(((pred-y)**2).mean())))
result=dict(records=len(rows),censored=sum(r['censored'] for r in rows),detected_by_assigned=sum(r['detected_by_assigned'] for r in rows),metrics=metrics,
 scope='Existing training episodes only. Cross-validation of observed outcomes, not counterfactual action value or independent development acceptance. Hidden detector events used only as supervised labels. No map, hidden target coordinates, or physical diagnostic poses used in input.',acceptance_claim=False)
(F/'public_latency_signal_diagnosis.json').write_text(json.dumps(dict(**result,rows=rows),indent=2))
print(json.dumps(result))
