"""CPU autograd only: initial fixed RLOO policy gradients and accepted parameter delta."""
from pathlib import Path
import collections, datetime, json, math, sys
import torch
O=Path(__file__).resolve().parent
R=O.parents[1]
sys.path[:0]=[str(O),str(R)]
from analyze_learning_signal import read,sha,guards,batch,load_model,fingerprint,action_mapping


def dot(a,b):return float(torch.dot(a,b))
def norm(a):return float(torch.linalg.vector_norm(a))
def cosine(a,b):
    den=norm(a)*norm(b)
    return dot(a,b)/den if den>0 else None


def main():
    torch.set_num_threads(1)
    plan=read(O/'plan.json');guards(plan)
    proof=read(O/'guarded/update_proof.json');signal=read(O/'learning_signal_result.json')
    assert proof['eligible_for_evaluation'] is True and proof['accepted_epochs']==1
    assert sha(R/proof['candidate'])==proof['candidate_sha256']==signal['candidate_sha256']
    parent,parent_cp=load_model(R/plan['behavior_checkpoint'])
    candidate,candidate_cp=load_model(R/proof['candidate'])
    assert fingerprint(parent_cp['model'])==proof['actual_behavior_fingerprint']==plan['behavior_state_fingerprint']
    named=[(name,p) for name,p in parent.named_parameters() if name.startswith(('ahgnn.','actor.'))]
    parameters=tuple(p for _,p in named)
    delta=torch.cat([(candidate_cp['model'][name]-p.detach()).reshape(-1).double() for name,p in named])
    blocks={};offset=0
    for name,p in named:
        block=name.split('.')[0];blocks.setdefault(block,[]).extend(range(offset,offset+p.numel()));offset+=p.numel()
    blocks={name:torch.tensor(indices,dtype=torch.long) for name,indices in blocks.items()}
    groups=collections.defaultdict(list);by_index={};files={}
    for task in plan['update_tasks']:
        folder=R/task['folder'];ep=folder/f"episode_{task['global_index']:03d}"
        cp=read(folder/'collection_proof.json');progress=read(folder/'progress.json')
        row=next(v for v in progress if v['episode']==task['global_index'])
        assert cp['source_unchanged'] is True and cp['optimizer_updates']==0 and row['behavior_sha256']==plan['behavior_state_fingerprint']
        assert all(cp[k]==task[k] for k in ('seed','global_index','sample_seed'))
        records=read(ep/'actor_inputs.json');assert len(records)==len(row['returns'])
        item={'task':task,'records':records,'completion':cp['alignment']['completion_seconds']}
        groups[task['seed']].append(item);by_index[task['index']]=item
        for f in (folder/'collection_proof.json',folder/'progress.json',ep/'actor_inputs.json'):files[str(f.relative_to(R))]=sha(f)
    count=sum(len(x['records']) for x in by_index.values());assert count==proof['records']==63
    gradients={};summaries={};max_replay=0.
    proof_groups={g['seed']:g for g in proof['groups']}
    for seed,items in groups.items():
        completions=[item['completion'] for item in items]
        advantages=[(sum(completions[:i]+completions[i+1:])/3-c)/15 for i,c in enumerate(completions)]
        if max(completions)-min(completions)<=1e-6:advantages=[0.]*4
        assert completions==proof_groups[seed]['completion_seconds']
        assert all(abs(a-b)<1e-12 for a,b in zip(advantages,proof_groups[seed]['advantages']))
        loss_terms=[];roles=collections.Counter()
        for item,advantage in zip(items,advantages):
            advantage=torch.tensor(advantage,dtype=torch.float32)
            item['advantage']=float(advantage);item['logratios']=[];item['initial_loss_terms']=[];item['candidate_loss_terms']=[]
            for record in item['records']:
                inputs=batch(record);action=int(record['action']);assert not inputs['action_mask'].flatten()[action]
                output=parent(**inputs)
                dist=torch.distributions.Categorical(logits=output['flat_logits'][0])
                newlog=dist.log_prob(torch.tensor(action));old=torch.tensor(record['log_prob'],dtype=torch.float32)
                max_replay=max(max_replay,abs(float(newlog.detach()-old)))
                ratio=(newlog-old).exp()
                term=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
                loss_terms.append(term);item['initial_loss_terms'].append(float(term.detach()))
                with torch.no_grad():
                    candidate_log=torch.log_softmax(candidate(**inputs)['flat_logits'][0],dim=-1)[action]
                    item['logratios'].append(float(candidate_log-newlog.detach()))
                    candidate_ratio=(candidate_log-old).exp()
                    item['candidate_loss_terms'].append(float(-torch.minimum(candidate_ratio*advantage,candidate_ratio.clamp(.8,1.2)*advantage)))
                ti=action//len(record['uav_ids']);roles[int(inputs['task_features'][0,ti,6:10].argmax())]+=1
        loss=torch.stack(loss_terms).sum()/count
        tensors=torch.autograd.grad(loss,parameters,allow_unused=True)
        gradient=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).double() for p,g in zip(parameters,tensors)])
        gradients[seed]=gradient
        summaries[seed]={'records':len(loss_terms),'transition_weight':len(loss_terms)/count,'roles_1_search_2_relay':dict(roles),
                         'fixed_episode_advantages':advantages,'initial_total63mean_loss_contribution':float(loss.detach()),
                         'weighted_gradient_norm':norm(gradient),'blocks':{name:{'gradient_norm':norm(gradient[idx])} for name,idx in blocks.items()}}
        del loss_terms,loss,tensors
    aggregate=sum(gradients.values());pairs=[]
    # Read-only decomposition under the accepted FIRST Adam step's fixed diagonal scaling.
    optimizer=candidate_cp['optimizer'];assert len(optimizer['param_groups'])==1
    pg=optimizer['param_groups'][0];assert pg['weight_decay']==0 and not pg.get('amsgrad',False)
    all_names=[name for name,_ in parent.named_parameters()]
    ids=dict(zip(all_names,pg['params']));denominators=[]
    for name,p in named:
        state=optimizer['state'][ids[name]];assert float(state['step'])==1
        denominators.append((state['exp_avg_sq'].double().sqrt()/math.sqrt(1-pg['betas'][1])+pg['eps']).reshape(-1))
    denominator=torch.cat(denominators)
    original_preclip_norm=2.3459014892578125
    clip_factor=min(1.,1./(original_preclip_norm+1e-6))
    adam_group_deltas={seed:-pg['lr']*clip_factor*g/denominator for seed,g in gradients.items()}
    reconstructed_delta=sum(adam_group_deltas.values())
    adam_decomposition={'meaning':'Fixed accepted first-step Adam diagonal denominator applied linearly to each initial CPU group gradient. Not separate per-group optimizer updates.',
        'learning_rate':pg['lr'],'gradient_clip_factor':clip_factor,
        'delta_reconstruction_l2_error':norm(reconstructed_delta-delta),
        'delta_reconstruction_relative_l2_error':norm(reconstructed_delta-delta)/norm(delta),
        'limits':'CPU gradients approximate CUDA gradients; retained float32 parameters introduce rounding. Adam denominator held fixed from original aggregate gradient.'}
    seeds=sorted(groups)
    for i,a in enumerate(seeds):
        for b in seeds[i+1:]:
            pairs.append({'seed_a':a,'seed_b':b,'dot':dot(gradients[a],gradients[b]),'cosine':cosine(gradients[a],gradients[b]),
                          'blocks':{name:{'dot':dot(gradients[a][idx],gradients[b][idx]),'cosine':cosine(gradients[a][idx],gradients[b][idx])} for name,idx in blocks.items()}})
    for seed,g in gradients.items():
        summaries[seed].update(cosine_with_aggregate_gradient=cosine(g,aggregate),
                              loss_directional_derivative_along_aggregate_descent=-dot(g,aggregate),
                              dot_with_actual_accepted_delta=dot(g,delta),
                              cosine_negative_gradient_with_actual_delta=cosine(-g,delta))
        for name,idx in blocks.items():summaries[seed]['blocks'][name]['dot_with_actual_accepted_delta']=dot(g[idx],delta[idx])
    trajectory_credit=[]
    signal_by_seed={g['seed']:g for g in signal['groups']}
    for seed,items in groups.items():
        episode_rows=[]
        for item in items:
            logratio=sum(item['logratios'])
            episode_rows.append({'episode_index':item['task']['index'],'completion_seconds':item['completion'],
                'decisions':len(item['records']),'fixed_advantage_float32':item['advantage'],
                'sum_logprob_candidate_minus_parent':logratio,'trajectory_likelihood_ratio':math.exp(logratio),
                'advantage_times_sum_logprob_change':item['advantage']*logratio,
                'initial_clipped_loss_sum':sum(item['initial_loss_terms']),
                'candidate_clipped_loss_sum':sum(item['candidate_loss_terms'])})
        lookup={x['episode_index']:x for x in episode_rows};selection=signal_by_seed[seed]
        fast=lookup[selection['fast_index']];slow=lookup[selection['slow_index']]
        credit=sum(x['advantage_times_sum_logprob_change'] for x in episode_rows)
        trajectory_credit.append({'seed':seed,'episodes':episode_rows,
            'fast_episode_index':fast['episode_index'],'slow_episode_index':slow['episode_index'],
            'fast_minus_slow_trajectory_relative_logodds_change':fast['sum_logprob_candidate_minus_parent']-slow['sum_logprob_candidate_minus_parent'],
            'sum_advantage_times_delta_logprob':credit,'original63mean_weighted_logcredit_change':credit/count,
            'initial63mean_clipped_policy_loss':sum(x['initial_clipped_loss_sum'] for x in episode_rows)/count,
            'candidate63mean_clipped_policy_loss':sum(x['candidate_clipped_loss_sum'] for x in episode_rows)/count,
            'clipped_policy_loss_change':sum(x['candidate_clipped_loss_sum']-x['initial_clipped_loss_sum'] for x in episode_rows)/count,
            'meaning':'Products of conditional GPPO action probabilities on each stored complete trajectory. Relative logodds compares change of fast versus slow trajectory likelihood; not a probability over task completion outcomes.'})
    action_probes=[]
    for group in signal['groups']:
        diagnostic=group.get('diagnostic')
        if diagnostic is None:continue
        item=by_index[group['fast_index']];record=item['records'][diagnostic['fast_record_index']]
        action=int(record['action']);inputs=batch(record)
        logp=torch.log_softmax(parent(**inputs)['flat_logits'][0],dim=-1)[action]
        tensors=torch.autograd.grad(logp,parameters,allow_unused=True)
        gp=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).double() for p,g in zip(parameters,tensors)])
        with torch.no_grad():newlogp=torch.log_softmax(candidate(**inputs)['flat_logits'][0],dim=-1)[action]
        action_probes.append({'seed':group['seed'],'fast_episode_index':group['fast_index'],'record_index':diagnostic['fast_record_index'],
                             'mapping':action_mapping(record,action),'initial_logprob_gradient_norm':norm(gp),
                             'actual_logprob_change':float(newlogp-logp.detach()),'first_order_logprob_change_along_actual_delta':dot(gp,delta),
                             'per_seed_raw_gradient_descent_logprob_derivatives':{str(seed):-dot(gp,g) for seed,g in gradients.items()},
                             'aggregate_raw_gradient_descent_logprob_derivative':-dot(gp,aggregate),
                             'per_seed_fixed_first_adam_logprob_contributions':{str(seed):dot(gp,d) for seed,d in adam_group_deltas.items()},
                             'sum_fixed_first_adam_logprob_contributions':dot(gp,reconstructed_delta),
                             'interpretation':'Per-seed raw gradient descent probes are local derivatives, not isolated Adam updates or physical action counterfactuals.'})
    assert max_replay<1e-4
    assert all(torch.equal(parent.state_dict()[name],tensor) for name,tensor in parent_cp['model'].items())
    guards(plan)
    result={'status':'COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'CPU autograd of original initial clipped policy-only loss; no optimizer, no new candidate, no transfer/evaluation outcomes read',
            'parent_sha256':plan['behavior_sha256'],'accepted_candidate_sha256':proof['candidate_sha256'],
            'parameter_scope':'All AHGNN encoder and actor parameters; critic excluded because policy-only coefficient0',
            'parameter_count':offset,'records':count,'normalization':'Each group sum divided by original total63; float32 advantages exactly as original update',
            'max_cpu_saved_logprob_error':max_replay,'trajectory_credit_primary':trajectory_credit,'group_gradients':summaries,'pairwise_gradient_comparisons':pairs,
            'aggregate_gradient_norm':norm(aggregate),'original_cuda_reported_preclip_norm':2.3459014892578125,
            'accepted_delta_norm':norm(delta),'aggregate_gradient_dot_delta':dot(aggregate,delta),'first_adam_fixed_diagonal_decomposition':adam_decomposition,
            'aggregate_descent_cosine_actual_delta':cosine(-aggregate,delta),'fast_action_local_probes':action_probes,
            'interpretation_rules':['Negative pairwise gradient dot/cosine is evidence of locally competing seed objectives on shared parameters','Negative group gradient dot actualdelta predicts local group loss improvement; positive predicts deterioration','Actual accepted update is clipped-gradient Adam; Euclidean aggregate descent andAdam direction need not match','A group-average loss improvement does not require every positivelyweighted trajectory action probability to increase','Primary fulltrajectory credit uses allcaptured decisions. Neither likelihoodchange nor surrogateimprovement establishes realclosed-loop outcomeimprovement','Action-specific local derivatives and finite logprobchanges do not establish physicalcausal effect on mission completion','These observations cannot establish uniquecause or explain historicalgeneralization regressions; Search/Relay samplecounts alone do not prove interference'],
            'input_file_sha256':files,'analysis_source_sha256':sha(Path(__file__)),'new_simulations':0,'optimizer_steps':0,'parameter_mutations':0}
    path=O/'gradient_conflict_audit.json'
    with path.open('w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['status','trajectory_credit_primary','aggregate_gradient_norm','first_adam_fixed_diagonal_decomposition']},indent=2))


if __name__=='__main__':main()
