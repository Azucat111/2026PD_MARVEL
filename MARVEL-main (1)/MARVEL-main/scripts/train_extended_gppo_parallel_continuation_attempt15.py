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
from audit_training_heading_control_contract import audit_record

def main():
 plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt15.json').read_text())
 for file,digest in {**plan['source_hashes'],**plan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
 pilot=ROOT/'artifacts/extended_parallel_rollout_execution_20261011_attempt2';assert json.loads((pilot/'pilot_execution_results.json').read_text())['passed']
 for name in ['pilot_30_static','pilot_60_dynamic']:
  assert json.loads((pilot/(name+'_math_audit.json')).read_text())['passed']
  assert json.loads((pilot/name/'rollout_only_execution_proof.json').read_text())['passed']
 prior=ROOT/plan['resume_from'];history=json.loads((prior/'progress.json').read_text())[:8];updates=json.loads((prior/'updates.json').read_text())[:1]
 assert len(history)==8 and len(updates)==1 and updates[0]['last_episode']==7
 out=ROOT/plan['out'];out.mkdir(exist_ok=False);(out/'workers').mkdir()
 assert hashlib.sha256((prior/'checkpoint_batch_000.pth').read_bytes()).hexdigest()==plan['resume_checkpoint_sha256']
 old_manifest=json.loads((prior/'manifest.json').read_text());args=SimpleNamespace(**old_manifest['args']);args.out=plan['out'];args.seeds=plan['training_seeds']
 manifest=dict(old_manifest);manifest['args']=vars(args);manifest['historical_initial8_source_hashes']=old_manifest['source_hashes'];manifest['source_hashes']=plan['source_hashes'];manifest['parallel_execution']=dict(worker_limit=plan['worker_limit'],same_frozen_behavior_per_batch=True,updates_reused_from_original_trainer=True,initial8_reused_without_simulation=True,replacement_seed_predeclared=True,pipeline_hashes=plan['pipeline_hashes'])
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'source_snapshot').mkdir()
 for rel in {**plan['source_hashes'],**plan['pipeline_hashes']}:
  if Path(rel).suffix in ('.py','.yaml','.png'):
   dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
 for i in range(8):shutil.copytree(prior/f'episode_{i:03d}',out/f'episode_{i:03d}')
 for name in ['behavior_batch_000.pth','checkpoint_batch_000.pth']:shutil.copyfile(prior/name,out/name)
 (out/'progress.json').write_text(json.dumps(history,indent=2));(out/'updates.json').write_text(json.dumps(updates,indent=2))
 cp=torch.load(out/'checkpoint_batch_000.pth',map_location='cuda',weights_only=False);model=GPPOActorCritic(edge_feature_dim=6).cuda();model.load_state_dict(cp['model']);model.eval();torch.set_num_threads(1)
 optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);optimizer.load_state_dict(cp['optimizer'])
 for group in optimizer.param_groups:group['lr']=args.lr
 collector=FrozenBehaviorBatch(8);update=existing_update_function();initial_hash=manifest['initial_sha256']
 def verify_sources():
  assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items())
  assert hashlib.sha256(Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth').read_bytes()).hexdigest()==manifest['frozen_checkpoint_sha256']
 def worker(index,behavior):
  seed=plan['training_seeds'][index];relative=str((out/'workers'/f'worker_{index:03d}').relative_to(ROOT))
  command=[sys.executable,'-u','-B','scripts/run_gppo_rollout_worker.py','--source-plan','artifacts/extended_integrated_training_plan_20261011_attempt15.json','--global-episode-index',str(index),'--training-wall-budget-seconds',str(plan['training_wall_budget_seconds']),'--initial',str(behavior),'--out',relative,'--seeds',str(seed),'--steps',str(args.steps),'--lr',str(args.lr),'--ppo-epochs',str(args.ppo_epochs),'--temperature',str(args.temperature),'--rollout-batch-size','8','--target-kl',str(args.target_kl),'--anchor-weight',str(args.anchor_weight),'--entropy-weight',str(args.entropy_weight),'--scenario',args.scenario,'--balanced-domains','--warning-rate-mps',str(args.warning_rate_mps),'--joint-rollout','--relay-reward-weight',str(args.relay_reward_weight)]
  with (out/'workers'/f'worker_{index:03d}.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  return index,result.returncode,out/'workers'/f'worker_{index:03d}'
 for start in range(8,64,8):
  verify_sources();collector.begin(model);batch=len(updates);behavior=out/f'behavior_batch_{batch:03d}.pth'
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
 assert len(history)==64 and len(updates)==8
 assert hashlib.sha256((ROOT/plan['initial']).read_bytes()).hexdigest()==initial_hash
 verify_sources();(out/'complete.json').write_text(json.dumps(dict(episodes=64,updates=8,initial_unchanged=True,first8_preserved=True,execution_only_parallel=True,acceptance_claim=False),indent=2))
if __name__=='__main__':main()
