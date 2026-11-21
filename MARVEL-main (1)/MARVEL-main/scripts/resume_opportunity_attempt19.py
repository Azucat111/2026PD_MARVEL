"""Continuation of an audited fixed on-policy cohort via isolated real rollouts."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,shutil,subprocess
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gppo_parallel_rollouts import load_captured_episode,validate_frozen_rows,existing_update_function,model_state_fingerprint
from gppo_rollout_batch import FrozenBehaviorBatch
from integrations.gppo.gppo_model import GPPOActorCritic
from audit_opportunity_inputs import audit_record

def main():
 plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261013_attempt19.json').read_text())
 for file,digest in {**plan['source_hashes'],**plan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
 out=ROOT/plan['out'];manifest=json.loads((out/'manifest.json').read_text());args=SimpleNamespace(**manifest['args'])
 history=json.loads((out/'progress.json').read_text());updates=json.loads((out/'updates.json').read_text())
 assert len(history)==8*len(updates) and len(updates)>0
 assert [r['episode'] for r in history]==list(range(len(history)))
 assert [r['seed'] for r in history]==plan['training_seeds'][:len(history)]
 stage=ROOT/'artifacts/extended_role_advantage_design_20261012_attempt1';sys.path.insert(0,str(stage))
 checkpoint=out/f'checkpoint_batch_{len(updates)-1:03d}.pth'
 cp=torch.load(checkpoint,map_location='cuda',weights_only=False);model=GPPOActorCritic(edge_feature_dim=9).cuda();model.load_state_dict(cp['model'],strict=True);model.eval();torch.set_num_threads(1)
 assert cp['episode']==len(history)-1
 optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);optimizer.load_state_dict(cp['optimizer'])
 for group in optimizer.param_groups:assert group['lr']==args.lr
 proof=dict(resumed_after=len(history),completed_episodes_reused=True,optimizer_restored=True,model_restored=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),execution_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),guarded_sources_unchanged=True)
 proof_path=out/f'resume_after_{len(history):03d}.json';proof_retry=0
 while proof_path.exists():proof_retry+=1;proof_path=out/f'resume_after_{len(history):03d}_retry_{proof_retry:02d}.json'
 proof_path.write_text(json.dumps(proof,indent=2));print('EXACT_CHECKPOINT_RESUME',json.dumps(proof),flush=True)
 collector=FrozenBehaviorBatch(8)
 from opportunity_training_source import training_source
 source=training_source();needle='advantages=(targets-old_values).detach()';assert source.count(needle)==1
 source=source.replace(needle,'from role_advantages import normalize_role_advantages\n            advantages=normalize_role_advantages((targets-old_values).detach(),[tr["role"] for tr in train])')
 update=existing_update_function(source);initial_hash=manifest['initial_sha256']
 def verify_sources():
  assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items())
  assert hashlib.sha256(Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth').read_bytes()).hexdigest()==manifest['frozen_checkpoint_sha256']
  assert hashlib.sha256((ROOT/'load_model/MARVEL/checkpoint.pth').read_bytes()).hexdigest()==manifest['native_checkpoint_sha256']
 def worker(index,behavior):
  seed=plan['training_seeds'][index];base=out/'workers'/f'worker_{index:03d}'
  choices=[base,*sorted((out/'workers').glob(f'worker_{index:03d}_retry_*'))]
  for folder in choices:
   proof_file=folder/'rollout_only_execution_proof.json'
   if proof_file.exists():
    p=json.loads(proof_file.read_text());assert p['passed'] and p['behavior_sha256']==collector.policy_sha256 and p['optimizer_updates']==0
    return index,0,folder
  folder=base
  retry=0
  while folder.exists() or Path(str(folder)+'.log').exists():retry+=1;folder=out/'workers'/f'worker_{index:03d}_retry_{retry:02d}'
  relative=str(folder.relative_to(ROOT))
  command=[sys.executable,'-u','-B','scripts/run_opportunity_rollout_worker.py','--source-plan','artifacts/extended_integrated_training_plan_20261013_attempt19.json','--global-episode-index',str(index),'--training-wall-budget-seconds',str(plan['training_wall_budget_seconds']),'--initial',str(behavior),'--out',relative,'--seeds',str(seed),'--steps',str(args.steps),'--lr',str(args.lr),'--ppo-epochs',str(args.ppo_epochs),'--temperature',str(args.temperature),'--rollout-batch-size','8','--target-kl',str(args.target_kl),'--anchor-weight',str(args.anchor_weight),'--entropy-weight',str(args.entropy_weight),'--scenario',args.scenario,'--balanced-domains','--warning-rate-mps',str(args.warning_rate_mps),'--joint-rollout','--relay-reward-weight',str(args.relay_reward_weight)]
  with Path(str(folder)+'.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  return index,result.returncode,folder
 for start in range(len(history),256,8):
  verify_sources();collector.begin(model);batch=len(updates);behavior=out/f'behavior_batch_{batch:03d}.pth'
  if behavior.exists():
   saved=torch.load(behavior,map_location='cpu',weights_only=False);assert saved['behavior_sha256']==collector.policy_sha256
   probe=GPPOActorCritic(edge_feature_dim=9);probe.load_state_dict(saved['model']);assert model_state_fingerprint(probe)==collector.policy_sha256
  else:
   torch.save(dict(protocol_version=manifest['protocol'],training_stage='online PPO',model=model.state_dict(),optimizer=optimizer.state_dict(),training_args=vars(args),reward_protocol=manifest['reward_protocol'],behavior_sha256=collector.policy_sha256),behavior)
  print('PARALLEL_BATCH_START',batch,start,collector.policy_sha256,flush=True)
  with ThreadPoolExecutor(max_workers=plan['worker_limit']) as executor:results=list(executor.map(lambda i:worker(i,behavior),range(start,start+8)))
  with (out/f'batch_{batch:03d}_execution.json').open('x') as f:json.dump([dict(index=i,exit_code=c,folder=str(folder.relative_to(ROOT))) for i,c,folder in results],f,indent=2)
  assert all(c==0 for _,c,_ in results),results
  verify_sources();rows=[]
  for index,code,folder in results:
   proof=json.loads((folder/'rollout_only_execution_proof.json').read_text());assert proof['passed'] and proof['behavior_sha256']==collector.policy_sha256 and proof['optimizer_updates']==0
   row,transitions=load_captured_episode(folder,index,'cuda');assert row['seed']==plan['training_seeds'][index]
   for t in transitions:audit_record(t['record'])
   collector.add_episode(model,transitions,row['returns']);rows.append(row)
   shutil.copytree(folder/f'episode_{index:03d}',out/f'episode_{index:03d}')
   row['behavior_checkpoint']=behavior.name;history.append(row)
  validate_frozen_rows(rows,collector.policy_sha256);assert collector.ready and collector.episodes==8
  (out/'progress.json').write_text(json.dumps(history,indent=2))
  update(model,optimizer,collector,args,out,start+7,updates,initial_hash,manifest['protocol'],manifest['reward_protocol'])
  verify_sources();print('PARALLEL_BATCH_COMPLETE',batch,flush=True)
 assert len(history)==256 and len(updates)==32
 assert hashlib.sha256((ROOT/plan['initial']).read_bytes()).hexdigest()==initial_hash
 verify_sources();(out/'complete.json').write_text(json.dumps(dict(episodes=256,updates=32,initial_unchanged=True,all_episodes_fresh=True,advantage_normalization=args.advantage_normalization,production_source_unchanged=True,acceptance_claim=False),indent=2))
if __name__=='__main__':main()
