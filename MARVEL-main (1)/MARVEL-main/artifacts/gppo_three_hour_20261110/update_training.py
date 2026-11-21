from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,datetime,time,argparse,collections
import torch
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R/'scripts'),str(R/'artifacts/extended_role_advantage_design_20261012_attempt1')]
from gppo_parallel_rollouts import load_captured_episode,validate_frozen_rows,existing_update_function,episode_domain
from gppo_rollout_batch import FrozenBehaviorBatch,model_state_fingerprint
from integrations.gppo.gppo_model import GPPOActorCritic
from aligned_reward import aligned_returns
from training_source import source
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);a=p.parse_args();B=O/'training'/f'batch_{a.batch:02d}';P=json.loads((B/'plan.json').read_text());torch.set_num_threads(1)
assert time.time()<datetime.datetime.fromisoformat(P['collection_deadline_utc']).timestamp()
dev=json.loads((O/'development/comparison.json').read_text());assert dev['status']=='COMPLETE'
assert all(v['success'] and v['safe'] for r in dev['rows'] for v in r['arms'].values())
for f,h in P['source_hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
execution=json.loads((B/'execution.json').read_text());assert len(execution)==8 and all(r.get('exit_code')==0 for r in execution)
cp=torch.load(R/P['behavior_checkpoint'],map_location='cuda',weights_only=False);model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(cp['model']);model.eval();assert model_state_fingerprint(model)==P['behavior_state_fingerprint']
collector=FrozenBehaviorBatch(8);collector.begin(model);rows=[];max_log_error=0.;max_value_error=0.;old_probs=[];old_actions=[];legal_masks=[];post_completion=0
for j,i in enumerate(P['global_indices']):
 folder=B/f'worker_{i:02d}';proof=json.loads((folder/'collection_proof.json').read_text());assert proof['full_h300'] and proof['optimizer_updates']==0
 row,transitions=load_captured_episode(folder,i,'cuda');n,d=episode_domain(i);assert row['seed']==P['training_seeds'][j] and row['uavs']==n and row['domain']==('dynamic' if d else 'static')
 trace=json.loads((folder/f'episode_{i:03d}'/'physical_joint_trace.json').read_text());labels=aligned_returns(trace,[tr['time'] for tr in transitions]);assert labels['returns']==row['returns'];post_completion+=sum(tr['time']>=labels['completion_seconds'] for tr in transitions)
 with torch.no_grad():
  for tr in transitions:
   out=model(**tr['batch']);dist=torch.distributions.Categorical(logits=out['flat_logits'][0]/P['update']['temperature']);max_log_error=max(max_log_error,abs(float(dist.log_prob(torch.tensor(tr['action'],device='cuda')))-tr['old_log_prob']));max_value_error=max(max_value_error,abs(float(out['value'][0])-tr['value']))
   old_probs.append(dist.probs.detach().clone());old_actions.append(int(out['flat_logits'].argmax()));legal_masks.append(~tr['batch']['action_mask'].flatten().bool())
 collector.add_episode(model,transitions,row['returns']);rows.append(row)
validate_frozen_rows(rows,collector.policy_sha256);assert max_log_error<1e-5 and max_value_error<1e-5;assert torch.tensor([tr['return'] for tr in collector.transitions]).std()>1e-8
transitions=list(collector.transitions);args=SimpleNamespace(**json.loads((B/f"worker_{P['global_indices'][0]:02d}"/'manifest.json').read_text())['args']);args.seeds=P['training_seeds'];args.advantage_normalization='existing_role_centered_standardized_v1'
out=B/'update';out.mkdir(exist_ok=False);args.out=str(out);updates=[];optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
if 'optimizer' in cp:optimizer.load_state_dict(cp['optimizer'])
for group in optimizer.param_groups:group['lr']=args.lr
s=source();needle='advantages=(targets-old_values).detach()';assert s.count(needle)==1;s=s.replace(needle,'from role_advantages import normalize_role_advantages\n            advantages=normalize_role_advantages((targets-old_values).detach(),[tr["role"] for tr in train])')
existing_update_function(s)(model,optimizer,collector,args,out,P['global_indices'][-1],updates,P['initial_sha256'],cp['protocol_version'],P['reward_protocol'])
assert len(updates)==1 and all(torch.isfinite(t).all() for t in model.state_dict().values());exact=[];changes=0
with torch.no_grad():
 for tr,old,old_action,legal in zip(transitions,old_probs,old_actions,legal_masks):
  output=model(**tr['batch']);dist=torch.distributions.Categorical(logits=output['flat_logits'][0]/args.temperature);p0=old[legal];p1=dist.probs[legal];exact.append(float((p0*(p0.clamp_min(1e-30).log()-p1.clamp_min(1e-30).log())).sum()));changes+=int(int(output['flat_logits'].argmax())!=old_action)
kl=max(e['approx_kl'] for e in updates[0]['epochs']);mean_exact=sum(exact)/len(exact);max_exact=max(exact);u=P['update'];eligible=kl<=u['accept_sampled_kl_at_most'] and mean_exact<=u['accept_mean_exact_kl_at_most'] and max_exact<=u['accept_max_state_exact_kl_at_most'];candidate=out/'checkpoint_batch_000.pth'
result=dict(status='COMPLETE',batch=a.batch,eligible_for_validation=eligible,eligible_as_next_behavior=eligible,episodes=8,records=len(transitions),role_counts=dict(collections.Counter(tr['role'] for tr in transitions)),optimizer_steps=len(updates[0]['epochs']),max_sampled_kl=kl,mean_exact_kl=mean_exact,max_state_exact_kl=max_exact,max_behavior_logprob_error=max_log_error,max_behavior_value_error=max_value_error,training_state_argmax_changes=changes,post_completion_decisions=post_completion,candidate=str(candidate.relative_to(R)),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),acceptance_claim=False)
(B/'update_proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

