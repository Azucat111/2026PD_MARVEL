from pathlib import Path
import sys,json,time,datetime,argparse,hashlib
R=Path.cwd();O=Path(__file__).resolve().parent;OLD=R/'artifacts/gppo_three_hour_20261110';sys.path[:0]=[str(OLD),str(R/'scripts'),str(R)]
from training_source import source
from aligned_reward import aligned_returns
import benchmark_fair_search as bench
from train_extended_gppo_wall_budget_attempt14 import bind_wall_budget
bench.run=bind_wall_budget(bench.run,1500.)
from gppo_parallel_rollouts import compile_rollout_source,load_captured_episode
import torch
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();P=json.loads((O/'plan.json').read_text());T=next(t for t in P['tasks'] if t['index']==a.index);i=T['global_index'];deadline=datetime.datetime.fromisoformat(P['collection_deadline_utc']).timestamp()
def check_deadline():
 if time.time()>=deadline:raise TimeoutError('Learnability collection deadline')
check_deadline()
assert hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()==(O/'plan_sha256.txt').read_text()
for f,h in P['controller_hashes'].items():assert hashlib.sha256((O/f).read_bytes()).hexdigest()==h,f
for f,h in P['source_hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
assert hashlib.sha256((R/P['behavior_checkpoint']).read_bytes()).hexdigest()==P['behavior_sha256']
s=source();assert s.count('action=dist.sample();flat=int(action)')==1
s=s.replace('action=dist.sample();flat=int(action)','action=torch.multinomial(dist.probs,1,generator=sampling_generator).squeeze(0);flat=int(action)')
ns={'__file__':str(R/'scripts/train_extended_gppo_batched.py'),'__name__':'isolated_repeated_seed_collection','aligned_returns':aligned_returns,'check_deadline':check_deadline}
exec(compile_rollout_source(s,i),ns)
ns['sampling_generator']=torch.Generator(device='cuda').manual_seed(T['sample_seed'])
# Omit pure diagnostic motion rendering/serialization; state/reward/actor capture remains unchanged.
ns['append_motion_frame']=lambda *a,**k:None;ns['capture_motion_frame']=lambda *a,**k:{}
u=P['update'];out=O/f'worker_{a.index:02d}'
sys.argv=[ns['__file__'],'--initial',str(R/P['behavior_checkpoint']),'--out',str(out),'--seeds',str(T['seed']),'--steps','300','--lr',str(u['lr']),'--ppo-epochs','4','--temperature','1','--rollout-batch-size','8','--target-kl','.005','--anchor-weight','0','--entropy-weight','0','--scenario',P['scenario'],'--balanced-domains','--warning-rate-mps','0.2','--joint-rollout','--relay-reward-weight','.25']
ns['main']();check_deadline();row,ts=load_captured_episode(out,i,'cpu');proof=json.loads((out/f'episode_{i:03d}/aligned_reward_proof.json').read_text());assert json.loads((out/'complete.json').read_text())['updates']==0
assert row['seed']==T['seed'] and row['uavs']==(30 if i%2==0 else 60)
assert row['domain']==('dynamic' if i%4>=2 else 'static')
assert hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()==(O/'plan_sha256.txt').read_text()
assert all(hashlib.sha256((O/f).read_bytes()).hexdigest()==h for f,h in P['controller_hashes'].items())
result=dict(**T,behavior_sha256=P['behavior_sha256'],behavior_state_fingerprint=P['behavior_state_fingerprint'],controller_hashes=P['controller_hashes'],alignment=proof,records=len(ts),heading_recorded=True,optimizer_updates=0,source_unchanged=all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in P['source_hashes'].items()))
(out/'collection_proof.json').write_text(json.dumps(result,indent=2));print('REPEAT_COMPLETE',json.dumps(result),flush=True)

