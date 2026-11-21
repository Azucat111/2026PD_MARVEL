from pathlib import Path
import sys,json,hashlib,collections,datetime
import torch
R=Path.cwd();O=Path(__file__).resolve().parent;OLD=R/'artifacts/gppo_three_hour_20261110';sys.path[:0]=[str(O),str(OLD),str(R),str(R/'scripts')]
from integrations.gppo.gppo_model import GPPOActorCritic
from gppo_rollout_batch import model_state_fingerprint
from gppo_parallel_rollouts import load_captured_episode
from aligned_reward import aligned_returns
from group_baseline import episode_advantages

def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def main():
 P=json.loads((O/'plan.json').read_text());C=json.loads((O/'update_contract.json').read_text());assert sha(O/'plan.json')==(O/'plan_sha256.txt').read_text()==C['plan_sha256'];assert sha(O/'update_contract.json')==(O/'update_contract_sha256.txt').read_text()
 for f,h in {**P['source_hashes'],**P['read_only_reference_hashes'],**C['source_hashes']}.items():assert sha(R/f)==h,f
 for f,h in P['controller_hashes'].items():assert sha(O/f)==h,f
 execution=json.loads((O/'execution.json').read_text());assert len(execution)==8 and all(r.get('exit_code')==0 and not r.get('terminated_by_budget') for r in execution)
 torch.set_num_threads(1);cp=torch.load(R/P['behavior_checkpoint'],map_location='cpu',weights_only=False);model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model']);model.eval();actual=model_state_fingerprint(model);assert actual==P['behavior_state_fingerprint'];assert sha(R/P['behavior_checkpoint'])==P['behavior_sha256']
 groups=collections.defaultdict(list);rows=[];maxlog=0.;maxvalue=0.;records=0
 for T in P['update_tasks']:
  D=R/T['folder'];proof=json.loads((D/'collection_proof.json').read_text());assert proof['source_unchanged'] is True and proof['optimizer_updates']==0
  assert all(proof[k]==T[k] for k in ['seed','global_index','sample_seed'])
  row,ts=load_captured_episode(D,T['global_index'],'cpu');assert row['behavior_sha256']==actual and row['seed']==T['seed']
  ep=D/f"episode_{T['global_index']:03d}";trace=json.loads((ep/'physical_joint_trace.json').read_text());assert len(trace)==3000
  alignment=aligned_returns(trace,[t['time'] for t in ts]);assert alignment==proof['alignment'] and row['returns']==alignment['returns'] and [t['return'] for t in ts]==alignment['returns']
  with torch.no_grad():
   for t in ts:
    output=model(**t['batch']);d=torch.distributions.Categorical(logits=output['flat_logits'][0]);assert not bool(t['batch']['action_mask'].flatten()[t['action']]);maxlog=max(maxlog,abs(float(d.log_prob(torch.tensor(t['action'])))-t['old_log_prob']));maxvalue=max(maxvalue,abs(float(output['value'][0])-t['value']))
  x=dict(task_index=T['index'],origin=T['origin'],seed=T['seed'],global_index=T['global_index'],sample_seed=T['sample_seed'],completion_seconds=alignment['completion_seconds'],success=alignment['success'],safe=alignment['whole_window_safe'],records=len(ts),initial=json.loads((ep/'gppo/initial.json').read_text()),config=json.loads((ep/'gppo/config.json').read_text()))
  groups[T['seed']].append(x);records+=len(ts)
 for seed,g in groups.items():
  assert len(g)==4 and all(x['initial']==g[0]['initial'] and x['config']==g[0]['config'] for x in g)
  cs=[x['completion_seconds'] for x in g];aa=episode_advantages(cs);var=max(cs)-min(cs)>1e-6
  if not var:aa=[0.]*4
  rows.append(dict(seed=seed,completion_seconds=cs,advantages=aa,has_variance=var,all_safe=all(x['safe'] for x in g),all_success=all(x['success'] for x in g),episode_records=[x['records'] for x in g],initial_and_config_identical=True,origins=[x['origin'] for x in g]))
 assert any(x['has_variance'] for x in rows)
 result=dict(status='COMPLETE_READINESS_CPU_CHECK',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),episodes=12,new_episodes=8,reference_episodes=4,records=records,groups=rows,max_cpu_logprob_replay_error=maxlog,max_cpu_value_replay_error=maxvalue,actual_behavior_fingerprint=actual,actual_behavior_file_sha256=sha(R/P['behavior_checkpoint']),source_hashes_checked=len(P['source_hashes']),reference_hashes_checked=len(P['read_only_reference_hashes']),controller_hashes_checked=len(P['controller_hashes']),contract_sha256=sha(O/'update_contract.json'),all_old_artifacts_unchanged=True,update_started_by_this_check=False,note='CPU replay is diagnostic; updater independently enforces GPU replay1e-5 gate before any optimizer step. Physical failures would remain included atC300; no success/safety filtering was applied.')
 (O/'collection_readiness.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
