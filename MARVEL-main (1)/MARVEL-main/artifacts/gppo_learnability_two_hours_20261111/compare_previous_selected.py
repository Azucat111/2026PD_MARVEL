"""Fixed CPU shadow comparison on saved CURRENT-candidate states; no closed-loop simulation."""
from pathlib import Path
import collections,datetime,hashlib,json,sys
import torch
O=Path(__file__).resolve().parent
R=O.parents[1]
sys.path[:0]=[str(O),str(R)]
from analyze_learning_signal import read,sha,batch,load_model,action_mapping

MODELS={
 'current':{'path':'artifacts/gppo_learnability_two_hours_20261111/guarded/update/checkpoint_batch_000.pth','sha256':'440d13d0ce33e1a68a87b38f57781280b58855bc75524f132c3f2ad6326fd7ae'},
 'previous_selected':{'path':'artifacts/gppo_three_hour_20261110/training/batch_00/update/checkpoint_batch_000.pth','sha256':'92415b8ff6f1a645045612b53f2716c130ccb6c0905c2874d8aa7ac4f6e1642d'},
 'parent17_behavior':{'path':'artifacts/gppo_three_hour_20261110/training/batch_00/behavior.pth','sha256':'e55eba9ca97a1dbfd9954a6360138a3768e3f997f74f92ffc441932c59db430a'},
}


@torch.no_grad()
def main():
    expected=[];missing=[];inputs={}
    for stage in ('evaluation','transfer'):
        plan_path=O/stage/'plan.json';plan=read(plan_path)
        assert sha(plan_path)==(O/stage/'protocol_sha256.txt').read_text().strip()
        assert plan['model_selection_permitted'] is False and plan['training_permitted'] is False
        assert plan['models']['candidate']['sha256']==MODELS['current']['sha256']
        assert plan['models']['parent']['sha256']==MODELS['parent17_behavior']['sha256']
        for source,digest in plan['source_hashes'].items():assert sha(R/source)==digest,source
        inputs[str(plan_path.relative_to(R))]=sha(plan_path)
        for index,case in enumerate(plan['cases']):
            base=O/stage/'data'/f"case_{index:02d}_{case['seed']}_{case['uavs']}_{'dynamic' if case['dynamic'] else 'static'}"/'candidate_original'
            record_path=base/'gppo/actor_inputs_9.json';complete_path=base/'complete.json'
            for path in (record_path,complete_path):
                if not path.exists():missing.append(str(path.relative_to(R)))
            expected.append((stage,index,case,record_path,complete_path))
    if missing:
        print(json.dumps({'status':'WAITING_FOR_ALL_CURRENT_CANDIDATE_TRAJECTORIES','missing':missing,'new_simulations':0,'optimizer_steps':0},indent=2))
        return
    torch.set_num_threads(1)
    models={}
    for name,info in MODELS.items():
        assert sha(R/info['path'])==info['sha256']
        models[name],_=load_model(R/info['path'])
    cases=[];stage_totals={}
    comparisons=[('current','previous_selected'),('current','parent17_behavior'),('previous_selected','parent17_behavior')]
    total_rows=0;replay_mismatches=[]
    for stage,index,case,record_path,complete_path in expected:
        complete=read(complete_path)
        assert complete['arm']=='candidate' and complete['case']==case and complete['mode']=='original'
        assert complete['checkpoint_sha256']==MODELS['current']['sha256']
        assert complete['source_stable'] is True and complete['optimizer_updates']==0
        assert complete['summary']['mission_steps']==300 and not complete['summary']['policy_execution_failures']
        records=read(record_path);details=[];counts=collections.Counter();role_counts=collections.Counter()
        for ri,record in enumerate(records):
            assert record['source_policy']=='candidate' and record['actual_model_edge_dim']==6
            x=batch(record);saved_edge_dim=x['edge_features'].shape[-1];assert saved_edge_dim==9
            x['edge_features']=x['edge_features'][...,:6].contiguous()
            recorded=int(record['action']);assert not x['action_mask'].flatten()[recorded]
            mapped=action_mapping(record,recorded)
            assert mapped['subtask_id']==record['chosen']['subtask_id'] and mapped['uav_id']==record['chosen']['uav_id']
            choices={};actions={}
            for name,model in models.items():
                logits=model(**x)['flat_logits'][0];action=int(logits.argmax());assert not x['action_mask'].flatten()[action]
                probs=logits.softmax(-1);actions[name]=action
                choices[name]={**action_mapping(record,action),'greedy_probability':float(probs[action]),'recorded_action_probability':float(probs[recorded])}
            exact_match=actions['current']==recorded
            if not exact_match:replay_mismatches.append({'stage':stage,'case':case,'record_index':ri,'recorded':mapped,'replayed':choices['current']})
            role=int(x['task_features'][0,recorded//len(record['uav_ids']),6:10].argmax());role_counts[role]+=1
            differences={f'{a}_vs_{b}':actions[a]!=actions[b] for a,b in comparisons}
            counts.update({key:int(value) for key,value in differences.items()})
            details.append({'record_index':ri,'time':record['time'],'step':record.get('step'),'role_1_search_2_relay':role,
                            'saved_edge_dim':saved_edge_dim,'actual_edge_dim':6,'recorded_current':mapped,
                            'current_replay_matches_recording':exact_match,'greedy_choices':choices,'differences':differences})
        total_rows+=len(records)
        stage_totals.setdefault(stage,{'records':0,**{f'{a}_vs_{b}':0 for a,b in comparisons}})
        stage_totals[stage]['records']+=len(records)
        for key,value in counts.items():stage_totals[stage][key]+=value
        cases.append({'stage':stage,'case_index':index,'case':case,'records':len(records),'role_counts':dict(role_counts),
                      'greedy_difference_counts':dict(counts),'all_current_vs_previous_choices_identical':counts['current_vs_previous_selected']==0,
                      'states':details})
        inputs[str(record_path.relative_to(R))]=sha(record_path);inputs[str(complete_path.relative_to(R))]=sha(complete_path)
    assert not replay_mismatches,('Current candidate CPU replay failed',replay_mismatches)
    for name,info in MODELS.items():assert sha(R/info['path'])==info['sha256']
    for path,digest in inputs.items():assert sha(R/path)==digest,path
    all_same=all(case['all_current_vs_previous_choices_identical'] for case in cases)
    result={'status':'COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':MODELS,
            'scope':'CPU shadow replay of three fixed models on CURRENT candidate trajectory states only. No previous-model rollout or model selection.',
            'all_current_recorded_actions_replayed_exactly':True,'total_current_trajectory_states':total_rows,
            'stage_totals':stage_totals,'cases':cases,'all_current_vs_previous_choices_identical':all_same,
            'conclusion_if_all_same':('These saved states exhibit no new greedy allocation behavior relative to the previous selected candidate. This small shadow test does not demonstrate additional behavioral benefit over that previous candidate.' if all_same else None),
            'limits':['Different shadow actions do not establish better or worse mission completion because later states remain from currentcandidate trajectory.',
                      'Matching allobservedactions is not a measured previouscandidate closed-loop score or general policy equivalence.',
                      'The30dynamic29.9s result must not be presented as a firstbreakthrough solely from comparison againstparent17; priorselectedcandidate history is a distinct comparison.',
                      'No training/evaluation protocol/model altered; no result-based selection; no freshsimulation oroptimizer.'],
            'input_file_sha256':inputs,'analysis_source_sha256':sha(Path(__file__)),'new_simulations':0,'optimizer_steps':0}
    target=O/'compare_previous_selected_result.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':'COMPLETE','output':str(target),'stage_totals':stage_totals,'all_current_vs_previous_choices_identical':all_same},indent=2))


if __name__=='__main__':main()
