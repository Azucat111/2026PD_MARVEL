"""Collect one real runtime episode with a frozen behavior policy, never PPO update."""
from pathlib import Path
import sys,argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gppo_parallel_rollouts import compile_rollout_source,episode_domain,load_captured_episode,model_state_fingerprint
from audit_opportunity_inputs import audit_record

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--global-episode-index',type=int,required=True);p.add_argument('--training-wall-budget-seconds',type=float,required=True)
 p.add_argument('--source-plan',default='artifacts/extended_integrated_training_plan_20261011_attempt14.json')
 opts,remaining=p.parse_known_args();assert len(remaining[remaining.index('--seeds')+1:remaining.index('--steps')])==1
 plan=json.loads((ROOT/opts.source_plan).read_text())
 assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in plan['source_hashes'].items())
 import benchmark_fair_search as bench
 from train_extended_gppo_wall_budget_attempt14 import bind_wall_budget
 original=bench.run;bench.run=bind_wall_budget(original,opts.training_wall_budget_seconds)
 ns={'__file__':str(ROOT/'scripts/train_extended_gppo_batched.py'),'__name__':'isolated_rollout_execution'}
 from opportunity_training_source import training_source
 source=training_source();exec(compile_rollout_source(source,opts.global_episode_index),ns)
 sys.argv=[ns['__file__'],*remaining]
 try:ns['main']()
 finally:bench.run=original
 out=ROOT/remaining[remaining.index('--out')+1];row,tr=load_captured_episode(out,opts.global_episode_index,'cpu')
 complete=json.loads((out/'complete.json').read_text());assert complete['updates']==0 and complete['initial_unchanged']
 n,dynamic=episode_domain(opts.global_episode_index);assert row['uavs']==n and row['domain']==('dynamic' if dynamic else 'static')
 errors=[audit_record(t['record']) for t in tr]
 assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in plan['source_hashes'].items())
 result=dict(passed=True,episode=opts.global_episode_index,seed=row['seed'],behavior_sha256=row['behavior_sha256'],optimizer_updates=0,records=len(tr),max_heading_feature_error=max(errors),source_hashes=plan['source_hashes'],execution_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),compiler_sha256=hashlib.sha256((ROOT/'scripts/gppo_parallel_rollouts.py').read_bytes()).hexdigest(),acceptance_claim=False)
 with (out/'rollout_only_execution_proof.json').open('x') as f:json.dump(result,f,indent=2)
 print('ISOLATED_FROZEN_ROLLOUT_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
