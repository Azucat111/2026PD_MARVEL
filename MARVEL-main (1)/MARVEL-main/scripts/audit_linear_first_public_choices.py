"""No-simulation first-state policy audit on already captured public inputs."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,json
import numpy as np,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from observed_find_cost_features import find_cost_features
from observed_matching_assignment import matching_assignment
from integrations.rule_scheduling import choose_rule_assignment
from integrations.gppo.task_graph import TaskType
D=ROOT/'artifacts/extended_public_visibility_preflight_20261015_attempt24';OUT=ROOT/'artifacts/extended_linear_find_cost_preflight_20261015_attempt29'
model=joblib.load(OUT/'linear_public8_alpha10.joblib')['estimator'];rows=[]
for seed in range(63400,63408):
 s=json.loads((D/'data'/f'seed_{seed}'/'public_state.json').read_text());raw=s['graph'];cfg=json.loads((D/'data'/f'seed_{seed}'/'rule/config.json').read_text());u=raw['uav_ids'];t=raw['task_ids']
 g=NS(num_uavs=len(u),num_subtasks=len(t),action_mask=np.asarray(raw['mask']),edge_features=np.asarray(raw['raw_edge_features'],dtype=np.float32),uav_states=[NS(uav_id=i) for i in u],subtasks=[NS(subtask_id=i,priority=raw['raw_task_features'][j][0],task_type=TaskType.TARGET_SEARCH) for j,i in enumerate(t)],flatten_action=lambda ti,ui:ti*len(u)+ui)
 baseline=choose_rule_assignment(g);pairs=np.argwhere(~g.action_mask);edge9=np.asarray(s['actor_batch']['edge_features'])[0];z=np.zeros(128)
 features=np.asarray([find_cost_features(raw,edge9,int(ti),int(ui),s['time'],cfg.get('dynamic_obstacles',[]),.1,z) for ti,ui in pairs]);pred=np.clip(model.predict(features),.1,150);cost=g.edge_features[:,:,2].copy()
 for (ti,ui),value in zip(pairs,pred):cost[ti,ui]=value
 g.edge_features=g.edge_features.copy();g.edge_features[:,:,2]=cost+1e-4*g.edge_features[:,:,2]
 choice=matching_assignment(g)
 rows.append(dict(seed=seed,baseline=dict(task=baseline.subtask_id,uid=baseline.uav_id),unfiltered_linear=dict(task=choice.subtask_id,uid=choice.uav_id),predicted_gain_seconds=float(cost[baseline.task_index,baseline.uav_index]-cost[choice.task_index,choice.uav_index])))
(OUT/'first_public_choice_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
