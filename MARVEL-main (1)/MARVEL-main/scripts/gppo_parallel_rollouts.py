"""Isolated rollout collection and unchanged existing-trainer PPO update reuse."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,sys,tempfile,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_extended_gppo_batched as trainer
from gppo_rollout_batch import FrozenBehaviorBatch,model_state_fingerprint
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.rule_scheduling import choose_rule_assignment

def episode_domain(index):
 if not isinstance(index,int) or index<0:raise ValueError('Nonnegative global episode index required')
 return (30 if index%2==0 else 60,index%4>=2)

def validate_frozen_rows(rows,fingerprint):
 if not rows or any(r['behavior_sha256']!=fingerprint for r in rows):raise ValueError('Mixed or missing behavior policy in batch')

def _collection_loop(tree):
 main=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'),None)
 if main is None:raise ValueError('Missing collection boundary')
 loops=[n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [getattr(e,'id',None) for e in n.target.elts]==['episode','seed']]
 if len(loops)!=1:raise ValueError('Missing or ambiguous collection boundary')
 return main,loops[0]

def compile_rollout_source(source,index):
 episode_domain(index)
 tree=ast.parse(source);main,loop=_collection_loop(tree)
 loop.iter=ast.parse(f'[({index},args.seeds[0])]',mode='eval').body
 updates=[n for n in loop.body if isinstance(n,ast.If) and 'collector.ready' in ast.unparse(n.test)]
 if len(updates)!=1:raise ValueError('Missing update boundary')
 at=loop.body.index(updates[0])
 loop.body[at:at+1]=ast.parse("torch.save(collector.transitions,out/'captured_transitions.pth')").body
 # Per-worker collection retains original samples/rewards and never updates model.
 main.body.insert(0,ast.parse('assert len(sys.argv)>1').body[0])
 ast.fix_missing_locations(tree)
 return compile(tree,str(ROOT/'scripts/train_extended_gppo_batched.py'),'exec')

def existing_update_function(source=None):
 source=source or (ROOT/'scripts/train_extended_gppo_batched.py').read_text()
 tree=ast.parse(source);_,loop=_collection_loop(tree)
 updates=[n for n in loop.body if isinstance(n,ast.If) and 'collector.ready' in ast.unparse(n.test)]
 if len(updates)!=1:raise ValueError('Missing update boundary')
 signature=ast.parse('def update(model,optimizer,collector,args,out,episode,updates,initial_hash,PROTOCOL,reward_protocol): pass').body[0]
 signature.body=copy.deepcopy(updates[0].body)+ast.parse('return update').body
 module=ast.fix_missing_locations(ast.Module(body=[signature],type_ignores=[]))
 ns=dict(vars(trainer));exec(compile(module,str(ROOT/'scripts/train_extended_gppo_batched.py'),'exec'),ns)
 return ns['update']

def load_captured_episode(folder,index,device):
 folder=Path(folder);history=json.loads((folder/'progress.json').read_text());row=next(r for r in history if r['episode']==index)
 ep=folder/f'episode_{index:03d}'
 saved=folder/'captured_transitions.pth'
 if saved.exists():
  transitions=torch.load(saved,map_location=device,weights_only=False)
 else:
  records=json.loads((ep/'actor_inputs.json').read_text());decisions=json.loads((ep/'gppo/decisions.json').read_text());transitions=[]
  for record,reward in zip(records,row['returns']):
   matches=[d for d in decisions if abs(d['current_time']-record['time'])<1e-9 and d['chosen']==record['chosen'] and d['uav_ids']==record['uav_ids'] and d['task_ids']==record['task_ids']]
   assert len(matches)==1
   d=matches[0];tasks=[SimpleNamespace(subtask_id=tid,priority=f[0]) for tid,f in zip(d['task_ids'],d['raw_task_features'])]
   uavs=[SimpleNamespace(uav_id=uid) for uid in d['uav_ids']]
   graph=SimpleNamespace(num_uavs=len(uavs),num_subtasks=len(tasks),action_mask=np.asarray(d['mask'],dtype=bool),edge_features=np.asarray(d['raw_edge_features']),subtasks=tasks,uav_states=uavs,flatten_action=lambda t,u:t*len(uavs)+u)
   action=record['action'];ti=action//len(uavs);role=int(np.argmax(d['raw_task_features'][ti][6:10]));teacher=choose_rule_assignment(graph).flat_action
   transitions.append(dict(role=role,batch={k:torch.as_tensor(v,device=device) for k,v in record['batch'].items()},action=action,old_log_prob=record['log_prob'],value=record['value'],time=record['time'],teacher=teacher,record=record,return_=reward))
   transitions[-1]['return']=transitions[-1].pop('return_')
 assert len(transitions)==len(row['returns'])
 return row,transitions

def _max_nested_error(a,b):
 if torch.is_tensor(a):return float((a.detach().cpu()-b.detach().cpu()).abs().max()) if a.numel() else 0.
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  return max((_max_nested_error(a[k],b[k]) for k in a),default=0.)
 if isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  return max((_max_nested_error(x,y) for x,y in zip(a,b)),default=0.)
 assert a==b,(a,b)
 return 0.

def replay_historical_update(folder,initial,device):
 folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());args=SimpleNamespace(**manifest['args'])
 cp=torch.load(initial,map_location=device,weights_only=False);model=GPPOActorCritic(edge_feature_dim=6).to(device);model.load_state_dict(cp['model']);model.eval()
 optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);optimizer.load_state_dict(cp['optimizer'])
 for g in optimizer.param_groups:g['lr']=args.lr
 collector=FrozenBehaviorBatch(8);collector.begin(model);rows=[]
 for i in range(8):
  row,tr=load_captured_episode(folder,i,device);rows.append(row);collector.add_episode(model,tr,row['returns'])
 validate_frozen_rows(rows,collector.policy_sha256)
 count=len(collector.transitions);updates=[]
 with tempfile.TemporaryDirectory() as temp:
  existing_update_function()(model,optimizer,collector,args,Path(temp),7,updates,manifest['initial_sha256'],trainer.PROTOCOL,manifest['reward_protocol'])
 expected=torch.load(folder/'checkpoint_batch_000.pth',map_location=device,weights_only=False)
 return dict(max_model_error=_max_nested_error(model.state_dict(),expected['model']),max_optimizer_error=_max_nested_error(optimizer.state_dict(),expected['optimizer']),transitions=count,old_updates=updates,passed=False)
