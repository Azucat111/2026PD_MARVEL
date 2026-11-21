"""Reconstruct extended inputs from saved public quantities only."""
import copy
from types import SimpleNamespace
import numpy as np
from integrations.gppo.opportunity_features import opportunity_extra
from integrations.gppo.task_graph import TaskType
from audit_training_heading_control_contract import audit_record as audit_heading

def capture_public_snapshot(graph,context):
 return dict(context=context,uavs=[dict(uav_id=int(u.uav_id),position=list(u.position),velocity=float(u.velocity),alive=bool(u.alive),available=bool(u.available),can_search=bool(u.can_search),assigned_task_num=int(u.assigned_task_num)) for u in graph.uav_states],tasks=[dict(subtask_id=int(t.subtask_id),task_type=int(t.task_type),position=None if t.position is None else list(t.position)) for t in graph.subtasks])

def audit_record(record):
 old=copy.deepcopy(record)
 old['batch']['edge_features']=np.asarray(record['batch']['edge_features'])[...,:6].tolist()
 heading_error=audit_heading(old)
 s=record['public_opportunity_snapshot']
 assert [u['uav_id'] for u in s['uavs']]==record['uav_ids']
 assert [t['subtask_id'] for t in s['tasks']]==record['task_ids']
 assert [u['position'] for u in s['uavs']]==record['public_heading_control_snapshot']['public_uav_positions']
 assert [t['position'] for t in s['tasks']]==record['public_heading_control_snapshot']['public_task_positions']
 graph=SimpleNamespace(num_uavs=len(s['uavs']),num_subtasks=len(s['tasks']),uav_states=[SimpleNamespace(**u) for u in s['uavs']],subtasks=[SimpleNamespace(**{**t,'task_type':TaskType(t['task_type'])}) for t in s['tasks']])
 expected=opportunity_extra(graph,s['context'])
 actual=np.asarray(record['batch']['edge_features'],dtype=np.float32)[0,:,:,6:]
 assert actual.shape==expected.shape
 error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
 assert error<1e-6,error
 return max(heading_error,error)

if __name__=='__main__':
 import argparse,json
 from pathlib import Path
 root=Path(__file__).resolve().parents[1]
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--out',required=True);args=p.parse_args()
 errors=[]
 for i in range(args.episodes):
  for record in json.loads((root/args.folder/f'episode_{i:03d}/actor_inputs.json').read_text()):errors.append(audit_record(record))
 assert errors
 result=dict(passed=True,episodes=args.episodes,records=len(errors),max_public_input_error=max(errors),production_control_clock_verified=True,truth_inputs=False,acceptance_claim=False)
 with (root/args.out).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
