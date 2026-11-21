"""Read-only offline review; no rollouts, optimizer steps or production changes."""
import sys,json,math,hashlib,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.gppo.gppo_model import GPPOActorCritic
F=ROOT/'artifacts/extended_gppo_joint_batch_20261013_attempt19'
OUT=ROOT/'artifacts/extended_system_review_20261014_attempt19'
torch.set_num_threads(1)
models={}
for name,file in [('warm',ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth'),('trained',F/'checkpoint_batch_031.pth')]:
 cp=torch.load(file,map_location='cpu',weights_only=False);m=GPPOActorCritic(edge_feature_dim=9);m.load_state_dict(cp['model']);m.eval();models[name]=m
results=[]
for folder in sorted((ROOT/'artifacts').glob('extended_integrated_candidate_9170_*_20261013_attempt19')):
 for arm in ['gppo','rule']:
  p=folder/arm
  if not (p/'summary.json').exists():continue
  summary=json.loads((p/'summary.json').read_text())
  decisions=json.loads((p/'decisions.json').read_text())
  records=json.loads((p/'actor_inputs.json').read_text())
  replay=[]
  for r in records:
   b={k:torch.tensor(v) for k,v in r['batch'].items()}
   match=[d for d in decisions if d['chosen']==r['chosen'] and abs(d['current_time']-r['time'])<1e-9]
   assert len(match)==1
   d=match[0];legal=[]
   for ti,row in enumerate(d['mask']):
    for ui,masked in enumerate(row):
     if not masked:legal.append((-d['raw_task_features'][ti][0],d['observed_total_time'][ti][ui],d['task_ids'][ti],d['uav_ids'][ui],ti*len(d['uav_ids'])+ui))
   teacher=min(legal)[-1]
   row=dict(time=r['time'],recorded=r['chosen'],teacher={'task':r['task_ids'][teacher//len(r['uav_ids'])],'uid':r['uav_ids'][teacher%len(r['uav_ids'])]})
   with torch.no_grad():
    for name,m in models.items():
     o=m(**b);logits=o['flat_logits'][0];a=int(logits.argmax());prob=torch.softmax(logits/2.5,0)
     row[name]=dict(task=r['task_ids'][a//len(r['uav_ids'])],uid=r['uav_ids'][a%len(r['uav_ids'])],teacher_agreement=a==teacher,recorded_agreement=a==r['action'],greedy_probability_at_training_temperature=float(prob[a]),entropy=float(torch.distributions.Categorical(logits=logits/2.5).entropy()),value=float(o['value'][0]))
   assert row['trained']['recorded_agreement']
   replay.append(row)
  results.append(dict(case=folder.name,arm=arm,metric=summary['metric'],joint=summary.get('secondary_joint_metric'),replay=replay))
history=json.loads((F/'progress.json').read_text());updates=json.loads((F/'updates.json').read_text())
reward=[];all_times=[];roles={};missing=[]
for ep in history:
 records=json.loads((F/f"episode_{ep['episode']:03d}"/'actor_inputs.json').read_text())
 times=ep['metric']['capped_detection_seconds'];h=ep['metric']['horizon_seconds']
 if ep['metric']['detected_targets']<ep['metric']['total_targets']:missing.append(dict(episode=ep['episode'],seed=ep['seed'],metric=ep['metric']))
 all_times.extend(times)
 for i,r in enumerate(records):
  role='search' if r['chosen']['subtask_id']<200000 else 'relay'
  search=sum(max(0,min(t,h)-r['time']) for t in times)/len(times)/h
  radio=.25*ep['relay_deficits'][i]
  reward.append(dict(role=role,search_penalty=search,radio_penalty=radio,return_value=ep['returns'][i],decision_time=r['time'],past_last_find=r['time']>=max(times)))
  roles[role]=roles.get(role,0)+1
stats={}
for role in roles:
 rr=[r for r in reward if r['role']==role]
 stats[role]=dict(count=len(rr),mean_search_penalty=statistics.mean(r['search_penalty'] for r in rr),mean_radio_penalty=statistics.mean(r['radio_penalty'] for r in rr),mean_abs_return=statistics.mean(abs(r['return_value']) for r in rr),past_last_find=sum(r['past_last_find'] for r in rr))
grad={}
for role in ['1','2']:
 vals=[u['role_stats'][role]['batch_weighted_policy_gradient_norm'] for u in updates if role in u['role_stats']]
 grad[role]=dict(mean=statistics.mean(vals),median=statistics.median(vals),maximum=max(vals))
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261013_attempt19.json').read_text())
source_mismatch=[f for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
report=dict(acceptance_claim=False,new_simulations=0,new_training=0,source_mismatch=source_mismatch,cases=results,reward_statistics=stats,weighted_role_gradient=grad,training=dict(episodes=len(history),transitions=len(reward),median_target_find_seconds=statistics.median(all_times),maximum_target_find_seconds=max(all_times),missed_target_episodes=missing),limits=['Matched-state warm replay is not a warm-policy closed-loop rollout','Radio/search reward correlations are not causal intervention','Single development seed; reserved final not evaluated'])
(OUT/'offline_review.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
for c in results:print(json.dumps(c))
