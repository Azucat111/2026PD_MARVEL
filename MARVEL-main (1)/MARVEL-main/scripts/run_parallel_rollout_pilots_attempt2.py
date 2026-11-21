from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/extended_parallel_rollout_execution_20261011_attempt2'
assert json.loads((folder/'unit_validation.json').read_text())['passed']
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt15.json').read_text());processes=[]
for name,index,seed in [('pilot_30_static',8,62510),('pilot_60_dynamic',11,62511)]:
 args=[sys.executable,'-u','-B','scripts/run_gppo_rollout_worker.py','--source-plan','artifacts/extended_integrated_training_plan_20261011_attempt15.json','--global-episode-index',str(index),'--training-wall-budget-seconds','1800','--initial',str(ROOT/plan['resume_from']/'checkpoint_batch_000.pth'),'--out',str(folder.relative_to(ROOT)/name),'--seeds',str(seed),'--steps','80','--lr','.0001','--ppo-epochs','4','--temperature','2.5','--rollout-batch-size','8','--target-kl','.03','--anchor-weight','.001','--entropy-weight','.005','--scenario',plan['scenario'],'--balanced-domains','--warning-rate-mps','.2','--joint-rollout','--relay-reward-weight','.25']
 log=(folder/(name+'.log')).open('x');proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);processes.append((name,proc,log));print('PILOT_START',name,proc.pid,flush=True)
results=[]
for name,proc,log in processes:
 code=proc.wait();log.close();results.append(dict(name=name,exit_code=code));print('PILOT_EXIT',name,code,flush=True)
 if code==0:
  subprocess.run([sys.executable,'-B','scripts/audit_joint_ppo_rollouts.py','--folder',str(folder.relative_to(ROOT)/name),'--episodes','1','--out',str(folder.relative_to(ROOT)/(name+'_math_audit.json'))],cwd=ROOT,check=True)
with (folder/'pilot_execution_results.json').open('x') as f:json.dump(dict(passed=all(r['exit_code']==0 for r in results),results=results,acceptance_claim=False),f,indent=2)
assert all(r['exit_code']==0 for r in results)
