"""One prespecified training-case PPO diagnostic; prepared only, never auto-run."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,datetime,collections,math
import torch
R=Path.cwd();G=Path(__file__).resolve().parent;O=G.parent;OLD=R/'artifacts/gppo_three_hour_20261110';sys.path[:0]=[str(G),str(O),str(OLD),str(R),str(R/'scripts')]
from group_baseline import episode_advantages
from epoch_transaction import EpochTransaction,NoAcceptedEpoch,state_fingerprint
from aligned_reward import aligned_returns
from training_source import source
from gppo_parallel_rollouts import load_captured_episode,existing_update_function
from gppo_rollout_batch import FrozenBehaviorBatch,model_state_fingerprint
from integrations.gppo.gppo_model import GPPOActorCritic

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module_sha(module):
 h=hashlib.sha256()
 for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def guards(P,C):
 assert sha(O/'update_contract.json')==(O/'update_contract_sha256.txt').read_text()
 assert sha(O/'plan.json')==(O/'plan_sha256.txt').read_text()==C['plan_sha256']
 for f,h in C['source_hashes'].items():assert sha(R/f)==h,f
 for f,h in P['source_hashes'].items():assert sha(R/f)==h,f
 for f,h in P['controller_hashes'].items():assert sha(O/f)==h,f
 for f,h in P['read_only_reference_hashes'].items():assert sha(R/f)==h,f
 assert sha(R/P['behavior_checkpoint'])==P['behavior_sha256']==C['behavior_sha256']
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(C['update_deadline_utc'])

def main():
 P=json.loads((O/'plan.json').read_text());C=json.loads((O/'update_contract.json').read_text());guards(P,C)
 GC=json.loads((G/'guarded_contract.json').read_text());assert sha(G/'guarded_contract.json')==(G/'guarded_contract_sha256.txt').read_text()
 assert all(sha(R/f)==h for f,h in GC['source_hashes'].items());assert sha(O/'update_proof.json')==GC['original_rejected_proof_sha256']
 assert sha(R/GC['original_rejected_candidate'])==GC['original_rejected_candidate_sha256']
 rejected_state=torch.load(R/GC['original_rejected_candidate'],map_location='cpu',weights_only=False)['model'];rejected_fingerprint=state_fingerprint(rejected_state)
 TLIST=P['update_tasks'];assert len(TLIST)==12 and len({t['index'] for t in TLIST})==12
 assert len(P['tasks'])==8 and len(P['reference_tasks'])==4 and C['max_policy_batches']==1
 assert collections.Counter(t['seed'] for t in TLIST)=={1880021:4,1280022:4,1880024:4}
 assert len({t['sample_seed'] for t in TLIST})==12
 execution=json.loads((O/'execution.json').read_text());assert len(execution)==8 and all(t.get('exit_code')==0 and not t.get('terminated_by_budget') for t in execution)
 torch.set_num_threads(1);torch.manual_seed(3981901)
 cp=torch.load(R/P['behavior_checkpoint'],map_location='cuda',weights_only=False);assert cp['reward_protocol']==P['reward_protocol']
 model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(cp['model']);model.eval();initial=model_state_fingerprint(model)
 assert initial==P['behavior_state_fingerprint'];critic_before=module_sha(model.critic)
 collector=FrozenBehaviorBatch(12);collector.begin(model);groups=collections.defaultdict(list);ordered=[];max_logerror=0.;max_valueerror=0.;states=[];oldprobs=[];metadata=[]
 for T in TLIST:
  D=R/T['folder'];proof=json.loads((D/'collection_proof.json').read_text())
  assert proof['source_unchanged'] is True and proof['optimizer_updates']==0
  assert all(proof[k]==T[k] for k in ['seed','global_index','sample_seed'])
  complete=json.loads((D/'complete.json').read_text());assert complete['updates']==0 and complete['initial_unchanged'] is True
  manifest=json.loads((D/'manifest.json').read_text());assert manifest['initial_sha256']==sha(R/P['behavior_checkpoint'])
  assert manifest['args']['temperature']==1. and manifest['args']['steps']==300 and manifest['args']['joint_rollout'] is True
  row,ts=load_captured_episode(D,T['global_index'],'cuda')
  assert row['seed']==T['seed'] and row['uavs']==(30 if T['global_index']%2==0 else 60)
  assert row['domain']==('dynamic' if T['global_index']%4>=2 else 'static')
  assert row['behavior_sha256']==initial and row['reward_protocol']==P['reward_protocol']
  ep=D/f"episode_{T['global_index']:03d}";trace=json.loads((ep/'physical_joint_trace.json').read_text());assert len(trace)==3000
  alignment=aligned_returns(trace,[t['time'] for t in ts]);assert alignment==proof['alignment']
  assert row['returns']==alignment['returns'] and [t['return'] for t in ts]==alignment['returns']
  summary=json.loads((ep/'gppo/summary.json').read_text());assert summary['mission_steps']==300 and summary['elapsed_simulation_seconds']==300. and not summary['policy_execution_failures']
  x=dict(task=T,folder=D,episode=ep,transitions=ts,completion=alignment['completion_seconds'],success=alignment['success'],safe=alignment['whole_window_safe'],actual_row_fingerprint=row['behavior_sha256'])
  groups[T['seed']].append(x);ordered.append(x)
 assert len(groups)==3
 advantages={};any_variance=False
 for seed,g in groups.items():
  assert len(g)==4
  inits=[json.loads((x['episode']/'gppo/initial.json').read_text()) for x in g];assert all(v==inits[0] for v in inits)
  configs=[json.loads((x['episode']/'gppo/config.json').read_text()) for x in g];assert all(v==configs[0] for v in configs)
  cs=[x['completion'] for x in g];aa=episode_advantages(cs);var=max(cs)-min(cs)>1e-6;any_variance=any_variance or var
  if not var:
   assert all(abs(a)<1e-6 for a in aa)
   aa=[0.]*len(g)
  for x,a in zip(g,aa):advantages[x['task']['index']]=a
  metadata.append(dict(seed=seed,completion_seconds=cs,advantages=aa,has_variance=var,full_h300=True,initial_identical=True,config_identical=True,success=[x['success'] for x in g],safe=[x['safe'] for x in g]))
 assert any_variance,'All three groups have zero variance; no policy update'
 flat_adv=[]
 for x in ordered:
  with torch.no_grad():
   for t in x['transitions']:
    output=model(**t['batch']);dist=torch.distributions.Categorical(logits=output['flat_logits'][0]);action=torch.tensor(t['action'],device='cuda')
    assert not bool(t['batch']['action_mask'].flatten()[t['action']])
    max_logerror=max(max_logerror,abs(float(dist.log_prob(action))-t['old_log_prob']));max_valueerror=max(max_valueerror,abs(float(output['value'][0])-t['value']))
    states.append(t);oldprobs.append(dist.probs.clone());flat_adv.append(advantages[x['task']['index']])
  collector.add_episode(model,x['transitions'],[t['return'] for t in x['transitions']])
 assert max_logerror<1e-5 and max_valueerror<1e-5 and collector.episodes==12
 flat_adv=torch.tensor(flat_adv,device='cuda');assert len(flat_adv)==len(collector.transitions)==len(states) and torch.isfinite(flat_adv).all()
 s=source();needle='advantages=(targets-old_values).detach()';assert s.count(needle)==1;s=s.replace(needle,'advantages=diagnostic_advantages.detach()')
 assert s.count('value=.25*')==1;s=s.replace('value=.25*','value=0.*')
 epoch_guard=EpochTransaction()
 def transactional_check(model,optimizer,epoch,sampled_kl):
  values=[]
  with torch.no_grad():
   for t,old in zip(states,oldprobs):
    q=model(**t['batch'])['flat_logits'][0].softmax(-1);legal=~t['batch']['action_mask'].flatten().bool();values.append(float((old[legal]*(old[legal].clamp_min(1e-30).log()-q[legal].clamp_min(1e-30).log())).sum()))
  metrics=dict(sampled_kl=sampled_kl,mean_exact_kl=sum(values)/len(values),max_state_exact_kl=max(values),attempted_model_fingerprint=model_state_fingerprint(model))
  metrics['matches_original_rejected_model']=metrics['attempted_model_fingerprint']==rejected_fingerprint
  accepted=epoch_guard.after_epoch(model,optimizer,epoch,metrics)
  if not accepted and epoch_guard.accepted_epochs==0:raise NoAcceptedEpoch('First proposed epoch breached unchanged caps; initial state restored')
  return accepted
 before='optimizer.zero_grad(set_to_none=True);loss=0.;policy_terms={role:[] for role in roles}'
 assert s.count(before)==1;s=s.replace(before,'epoch_guard.before_epoch(model,optimizer,epoch)\n                '+before)
 after='epochs.append(dict(epoch=epoch,approx_kl=kl,clip_fraction=clipped,preclip_gradient_norm=gradnorm))'
 assert s.count(after)==1;s=s.replace(after,after+'\n                accepted=transactional_check(model,optimizer,epoch,kl)\n                epochs[-1]["transaction_accepted"]=accepted\n                epochs[-1]["rolled_back"]=not accepted\n                if not accepted:break')
 f=existing_update_function(s);f.__globals__.update(diagnostic_advantages=flat_adv,epoch_guard=epoch_guard,transactional_check=transactional_check)
 args=SimpleNamespace(**json.loads((ordered[0]['folder']/'manifest.json').read_text())['args']);args.seeds=[t['seed'] for t in TLIST];args.ppo_epochs=4;args.lr=2e-5;args.target_kl=.005;args.temperature=1.;args.entropy_weight=0.;args.anchor_weight=0.
 guards(P,C);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);outdir=G/'update';outdir.mkdir(exist_ok=False);updates=[];args.out=str(outdir)
 try:
  f(model,optimizer,collector,args,outdir,11,updates,P['initial_sha256'],cp['protocol_version'],P['reward_protocol'])
 except NoAcceptedEpoch:
  assert model_state_fingerprint(model)==initial and epoch_guard.accepted_epochs==0 and not list(outdir.glob('checkpoint*.pth'))
  proof=dict(status='COMPLETE',eligible_for_evaluation=False,reason='First epoch rejected; model and optimizer restored to initial; no evaluable candidate',candidate=None,candidate_sha256=None,accepted_epochs=0,attempted_epochs=len(epoch_guard.history),epoch_transactions=epoch_guard.history,actual_behavior_fingerprint=initial,actual_behavior_file_sha256=sha(R/P['behavior_checkpoint']),all_rows_match_actual_behavior=True,source_unchanged=True,records=len(states),groups=metadata,contract_sha256=sha(G/'guarded_contract.json'))
  (G/'update_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2));return
 except BaseException:
  epoch_guard.rollback(model,optimizer);raise
 assert len(updates)==1 and all(torch.isfinite(v).all() for v in model.state_dict().values());assert module_sha(model.critic)==critic_before
 exact=[];flips=0
 with torch.no_grad():
  for t,old in zip(states,oldprobs):
   q=model(**t['batch'])['flat_logits'][0].softmax(-1);legal=~t['batch']['action_mask'].flatten().bool();exact.append(float((old[legal]*(old[legal].clamp_min(1e-30).log()-q[legal].clamp_min(1e-30).log())).sum()));flips+=int(int(q.argmax())!=int(old.argmax()))
 mean=sum(exact)/len(exact);maxexact=max(exact);sampled=max(e['sampled_kl'] for e in epoch_guard.history if e['accepted']);eligible=epoch_guard.accepted_epochs>0 and all(math.isfinite(v) for v in [mean,maxexact,sampled]) and mean<=.01 and maxexact<=.05 and sampled<=.01
 guards(P,C);candidate=outdir/'checkpoint_batch_000.pth'
 proof=dict(status='COMPLETE',eligible_for_evaluation=eligible,groups=metadata,episodes=12,new_episodes=8,read_only_reference_episodes=4,records=len(states),optimizer_steps=epoch_guard.accepted_epochs,accepted_epochs=epoch_guard.accepted_epochs,attempted_epochs=len(epoch_guard.history),epoch_transactions=epoch_guard.history,rollback_performed=any(e['rolled_back'] for e in epoch_guard.history),original_rejected_model_fingerprint=rejected_fingerprint,actual_behavior_fingerprint=initial,actual_behavior_file_sha256=sha(R/P['behavior_checkpoint']),all_rows_match_actual_behavior=True,source_unchanged=True,critic_parameters_unchanged=True,max_logprob_replay_error=max_logerror,max_value_replay_error=max_valueerror,mean_exact_kl=mean,max_state_exact_kl=maxexact,max_sampled_kl=sampled,training_state_greedy_changes=flips,candidate=str(candidate.relative_to(R)),candidate_sha256=sha(candidate),contract_sha256=sha(G/'guarded_contract.json'),scope='Guarded reproducible replay of the same original batch; per-epoch model+Adam rollback on unchanged KL caps; not result-based model selection. One training-case within-seed leave-one-out baseline PPO diagnostic; zero-variance groups retained A0; critic loss0; fixed scale0.05; original transition-mean weighting; no acceptance/generalization/model-selection claim')
 (G/'update_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
