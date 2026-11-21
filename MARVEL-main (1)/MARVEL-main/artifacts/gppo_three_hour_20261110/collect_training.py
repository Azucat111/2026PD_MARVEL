from pathlib import Path
import sys,json,time,datetime,argparse,hashlib
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R/'scripts')]
from training_source import source
from aligned_reward import aligned_returns
from gppo_parallel_rollouts import compile_rollout_source,load_captured_episode,episode_domain
from audit_training_heading_control_contract import audit_record
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();B=O/'training'/f'batch_{a.batch:02d}';P=json.loads((B/'plan.json').read_text());assert a.index in P['global_indices']
deadline=datetime.datetime.fromisoformat(P['collection_deadline_utc']).timestamp()
def check_deadline():
 if time.time()>=deadline:raise TimeoutError('Absolute collection budget exhausted')
check_deadline()
for f,h in P['source_hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
assert hashlib.sha256((R/P['behavior_checkpoint']).read_bytes()).hexdigest()==P['behavior_sha256']
import benchmark_fair_search as bench
from train_extended_gppo_wall_budget_attempt14 import bind_wall_budget
bench.run=bind_wall_budget(bench.run,900.)
out=B/f'worker_{a.index:02d}';ns={'__file__':str(R/'scripts/train_extended_gppo_batched.py'),'__name__':'isolated_corrected_collection','aligned_returns':aligned_returns,'check_deadline':check_deadline}
exec(compile_rollout_source(source(),a.index),ns);seed=P['training_seeds'][P['global_indices'].index(a.index)];u=P['update']
sys.argv=[ns['__file__'],'--initial',str(R/P['behavior_checkpoint']),'--out',str(out),'--seeds',str(seed),'--steps','300','--lr',str(u['lr']),'--ppo-epochs',str(u['ppo_epochs']),'--temperature',str(u['temperature']),'--rollout-batch-size','8','--target-kl',str(u['target_kl']),'--anchor-weight','0','--entropy-weight','0','--scenario',P['scenario'],'--balanced-domains','--warning-rate-mps','0.2','--joint-rollout','--relay-reward-weight','0.25']
ns['main']();row,tr=load_captured_episode(out,a.index,'cpu');n,dynamic=episode_domain(a.index);assert row['uavs']==n and row['domain']==('dynamic' if dynamic else 'static')
errors=[audit_record(t['record']) for t in tr];proof=json.loads((out/f'episode_{a.index:03d}'/'aligned_reward_proof.json').read_text());assert row['returns']==proof['returns'];assert json.loads((out/'complete.json').read_text())['updates']==0
(out/'collection_proof.json').write_text(json.dumps(dict(index=a.index,seed=seed,uavs=n,dynamic=dynamic,records=len(tr),heading_error=max(errors),full_h300=True,optimizer_updates=0,alignment=proof),indent=2));print('CORRECTED_COLLECTION_COMPLETE',a.index,proof['completion_seconds'],proof['success'],len(tr),flush=True)
