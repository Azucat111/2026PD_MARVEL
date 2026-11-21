"""CPU-only fixed training-trajectory diagnostic; never trains or reads transfer outcomes."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, sys
import torch

O=Path(__file__).resolve().parent
R=O.parents[1]
sys.path.insert(0,str(R))
from integrations.gppo.gppo_model import GPPOActorCritic


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fingerprint(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        tensor=tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch(record):
    return {key:torch.as_tensor(value,dtype=torch.bool if key.endswith('mask') or key.endswith('matrix') else torch.float32)
            for key,value in record['batch'].items()}


def input_key(record):
    digest=hashlib.sha256()
    digest.update(json.dumps([record['uav_ids'],record['task_ids']],separators=(',',':')).encode())
    for key,tensor in sorted(batch(record).items()):
        digest.update(key.encode());digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def same_input(a,b):
    return (a['uav_ids']==b['uav_ids'] and a['task_ids']==b['task_ids']
            and a['batch'].keys()==b['batch'].keys()
            and all(torch.equal(batch(a)[k],batch(b)[k]) for k in a['batch']))


def action_mapping(record,action):
    ti,ui=divmod(int(action),len(record['uav_ids']))
    return {'flat_action':int(action),'task_index':ti,'uav_index':ui,
            'subtask_id':record['task_ids'][ti],'uav_id':record['uav_ids'][ui]}


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['model']['ahgnn.edge_input.weight'].shape[1]==6
    model=GPPOActorCritic(edge_feature_dim=6).cpu()
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    return model,checkpoint


def guards(plan):
    assert sha(O/'plan.json')==(O/'plan_sha256.txt').read_text()
    for name,digest in plan['source_hashes'].items():assert sha(R/name)==digest,name
    for name,digest in plan['controller_hashes'].items():assert sha(O/name)==digest,name
    for name,digest in plan['read_only_reference_hashes'].items():assert sha(R/name)==digest,name
    assert sha(R/plan['behavior_checkpoint'])==plan['behavior_sha256']


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof',type=Path,default=O/'guarded/update_proof.json')
    args=parser.parse_args()
    proof_path=args.proof.resolve()
    assert proof_path.is_relative_to(O.resolve()),'Proof must belong to current diagnostic directory'
    if not proof_path.exists():
        print(json.dumps({'status':'WAITING_FOR_UPDATE_PROOF','new_simulations':0,'optimizer_steps':0}))
        return
    plan=read(O/'plan.json');proof=read(proof_path);guards(plan)
    if proof.get('eligible_for_evaluation') is not True:
        print(json.dumps({'status':'REJECTED_PROOF_NOT_ANALYZED','proof':str(proof_path),
                          'reason':'Candidate failed declared update gates; never treat as eligible',
                          'new_simulations':0,'optimizer_steps':0}))
        return
    assert proof['status']=='COMPLETE' and proof['source_unchanged'] is True
    assert proof['all_rows_match_actual_behavior'] is True
    assert proof['actual_behavior_file_sha256']==plan['behavior_sha256']
    parent,parent_checkpoint=load_model(R/plan['behavior_checkpoint'])
    parent_fp=fingerprint(parent_checkpoint['model'])
    assert parent_fp==plan['behavior_state_fingerprint']==proof['actual_behavior_fingerprint']
    candidate_path=(R/proof['candidate']).resolve()
    assert candidate_path.is_relative_to(O.resolve()) and sha(candidate_path)==proof['candidate_sha256']
    candidate,_=load_model(candidate_path)
    torch.set_num_threads(1)
    groups=collections.defaultdict(list);rows=[];max_replay_error=0.
    assert len(plan['update_tasks'])==12
    source_files={str(proof_path.relative_to(R)):sha(proof_path)}
    for task in plan['update_tasks']:
        folder=R/task['folder'];ep=folder/f"episode_{task['global_index']:03d}"
        cp=read(folder/'collection_proof.json');complete=read(folder/'complete.json')
        manifest=read(folder/'manifest.json');progress=read(folder/'progress.json')
        actual=next(row for row in progress if row['episode']==task['global_index'])
        assert all(cp[k]==task[k] for k in ('seed','global_index','sample_seed'))
        assert cp['source_unchanged'] is True and cp['optimizer_updates']==0 and complete['updates']==0
        assert complete['initial_unchanged'] is True and manifest['initial_sha256']==plan['behavior_sha256']
        assert manifest['args']['temperature']==1 and manifest['args']['steps']==300
        assert actual['behavior_sha256']==parent_fp and actual['seed']==task['seed']
        behavior=torch.load(folder/'behavior_batch_000.pth',map_location='cpu',weights_only=True)
        assert fingerprint(behavior['model'])==parent_fp
        summary=read(ep/'gppo/summary.json');alignment=read(ep/'aligned_reward_proof.json')
        assert summary['mission_steps']==300 and summary['elapsed_simulation_seconds']==300.
        assert summary['valid_for_comparison'] and not summary['policy_execution_failures']
        assert alignment==cp['alignment'] and len(read(ep/'physical_joint_trace.json'))==3000
        records=read(ep/'actor_inputs.json');assert len(records)==len(actual['returns'])
        for record in records:
            x=batch(record);action=int(record['action'])
            assert not x['action_mask'].flatten()[action]
            mapping=action_mapping(record,action)
            assert mapping['subtask_id']==record['chosen']['subtask_id'] and mapping['uav_id']==record['chosen']['uav_id']
            dist=torch.distributions.Categorical(logits=parent(**x)['flat_logits'][0])
            max_replay_error=max(max_replay_error,abs(float(dist.log_prob(torch.tensor(action)))-record['log_prob']))
        item={'task':task,'completion_seconds':alignment['completion_seconds'],'success':alignment['success'],
              'whole_window_safe':alignment['whole_window_safe'],'records':records,'initial':read(ep/'gppo/initial.json'),
              'config':read(ep/'gppo/config.json')}
        groups[task['seed']].append(item)
        for path in (folder/'collection_proof.json',folder/'progress.json',ep/'actor_inputs.json',ep/'aligned_reward_proof.json'):
            source_files[str(path.relative_to(R))]=sha(path)
    assert max_replay_error<1e-4,'CPU-vs-savedGPU replay exceeds diagnostic tolerance'
    for seed,episodes in sorted(groups.items()):
        assert len(episodes)==4
        assert all(x['initial']==episodes[0]['initial'] and x['config']==episodes[0]['config'] for x in episodes)
        ordered=sorted(episodes,key=lambda x:(x['completion_seconds'],x['task']['index']))
        fast,slow=ordered[0],ordered[-1]
        result={'seed':seed,'selection_rule':'min/max complete trajectory completion_seconds; ties by fixed task index',
                'all_episode_outcomes':[{'index':x['task']['index'],'completion_seconds':x['completion_seconds'],
                    'success':x['success'],'whole_window_safe':x['whole_window_safe']} for x in ordered],
                'fast_index':fast['task']['index'],'slow_index':slow['task']['index'],
                'has_outcome_variation':slow['completion_seconds']-fast['completion_seconds']>1e-6,
                'initial_public_state_and_config_identical':True}
        slow_by_key=collections.defaultdict(list)
        for j,record in enumerate(slow['records']):slow_by_key[(record['time'],input_key(record))].append((j,record))
        matches=[]
        for i,a in enumerate(fast['records']):
            for j,b in slow_by_key[(a['time'],input_key(a))]:
                if a['action']!=b['action'] and same_input(a,b):matches.append((a['time'],i,j,a,b))
        if not matches:
            result.update(status='NO_SAME_INPUT_ACTION_DIVERGENCE',diagnostic=None)
        else:
            time,i,j,a,b=min(matches,key=lambda m:m[:3]);x=batch(a)
            old_logits=parent(**x)['flat_logits'][0];new_logits=candidate(**x)['flat_logits'][0]
            old_log=torch.log_softmax(old_logits,dim=-1);new_log=torch.log_softmax(new_logits,dim=-1)
            p=old_log.exp();q=new_log.exp();legal=~x['action_mask'].flatten()
            kl=float((p[legal]*(old_log[legal]-new_log[legal])).sum())
            fa,sa=int(a['action']),int(b['action'])
            result.update(status='SAME_INPUT_ACTION_DIVERGENCE_FOUND',diagnostic={
                'time':time,'fast_record_index':i,'slow_record_index':j,'same_input_sha256':input_key(a),
                'matching_policy':'All batch tensors bitwise-equal inCPU dtype, UAV/taskIDs equal, decisiontime equal',
                'fast_trajectory_action':{**action_mapping(a,fa),'parent_probability':float(p[fa]),'candidate_probability':float(q[fa]),'delta':float(q[fa]-p[fa])},
                'slow_trajectory_action':{**action_mapping(b,sa),'parent_probability':float(p[sa]),'candidate_probability':float(q[sa]),'delta':float(q[sa]-p[sa])},
                'parent_argmax':action_mapping(a,int(p.argmax())),
                'candidate_argmax':action_mapping(a,int(q.argmax())),
                'exact_kl_parent_to_candidate':kl,
                'relative_fast_over_slow_logodds_change':float((new_log[fa]-new_log[sa])-(old_log[fa]-old_log[sa])),
                'better_trajectory_action_probability_increased':bool(q[fa]>p[fa]) if result['has_outcome_variation'] else None})
        result['causal_limit']='Whole-trajectory credit association only; first divergent action is not proven an isolated cause. Equal-outcome groups have no better trajectory label.'
        rows.append(result)
    guards(plan)
    assert sha(candidate_path)==proof['candidate_sha256']
    result={'status':'COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Prespecified training trajectories only; no transfer outcomes read; no model selection, simulation or optimizer',
            'proof_path':str(proof_path.relative_to(R)),'proof_sha256':sha(proof_path),
            'parent_sha256':plan['behavior_sha256'],'candidate_sha256':proof['candidate_sha256'],
            'candidate_eligible_for_evaluation':proof['eligible_for_evaluation'],'actual_behavior_fingerprint':parent_fp,
            'all12_actual_behavior_models_identical':True,'max_cpu_old_logprob_replay_error':max_replay_error,
            'cpu_replay_tolerance':1e-4,'groups':rows,'input_file_sha256':source_files,
            'analysis_source_sha256':sha(Path(__file__)),'new_simulations':0,'optimizer_steps':0}
    destination=O/'learning_signal_result.json'
    with destination.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({'status':'COMPLETE','output':str(destination),'groups':rows},indent=2))


if __name__=='__main__':main()
