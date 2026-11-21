"""Contracts for independent extended-policy training, not production inference."""
import copy
import pytest
import torch
from scripts.train_extended_gppo import example_from_record, validate_seed_split, train_step
from integrations.gppo.gppo_model import GPPOActorCritic

def record():
    return dict(uav_ids=[9,2],task_ids=[12,5],candidate_budget=0,
        mask=[[False,True],[False,False]],
        raw_task_features=[[2]+[0]*11,[1]+[0]*11],
        raw_edge_features=[[[8,0,8],[0,0,0]],[[1,0,1],[2,0,2]]],
        model_uav_features=[[0]*13,[0]*13],model_task_features=[[0]*12,[0]*12],
        model_edge_features=[[[.8,0,.8],[0,0,0]],[[.1,0,.1],[.2,0,.2]]],
        capability_matrix=[[True,True],[True,True]],precedence_matrix=[[False,False],[False,False]])

def test_priority_precedes_cost_and_invalid_zero_cost_never_wins():
    ex=example_from_record(record(),'cpu')
    assert ex['label']==0

def test_ids_break_equal_cost_ties_not_array_order():
    r=record();r['raw_task_features'][0][0]=1;r['raw_edge_features'][0][0][2]=1
    r['raw_edge_features'][1][1][2]=1
    assert example_from_record(r,'cpu')['label']==3

def test_all_invalid_graph_is_skipped():
    r=record();r['mask']=[[True,True],[True,True]]
    assert example_from_record(r,'cpu') is None

def test_nonfinite_or_filtered_observations_rejected():
    r=record();r['model_edge_features'][0][0][0]=float('nan')
    with pytest.raises(ValueError):example_from_record(r,'cpu')
    r=record();r['candidate_budget']=4
    with pytest.raises(ValueError):example_from_record(r,'cpu')

def test_seed_overlap_is_rejected():
    with pytest.raises(ValueError):validate_seed_split([1],[1],[3])
    validate_seed_split([1],[2],[3])

def test_real_optimizer_learns_without_mutating_initial_state():
    torch.manual_seed(7);torch.set_num_threads(1)
    model=GPPOActorCritic(embed_dim=16,n_layers=1);model.eval()
    initial=copy.deepcopy(model.state_dict())
    ex=example_from_record(record(),'cpu');optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    before=torch.nn.functional.cross_entropy(model(**ex['batch'])['flat_logits'],torch.tensor([0])).item()
    for _ in range(20):train_step(model,optimizer,ex)
    after=torch.nn.functional.cross_entropy(model(**ex['batch'])['flat_logits'],torch.tensor([0])).item()
    assert after<before*.5
    assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())

def test_metrics_are_json_serializable():
    import json
    from scripts.train_extended_gppo import evaluate
    model=GPPOActorCritic(embed_dim=16,n_layers=1)
    json.dumps(evaluate(model,[example_from_record(record(),'cpu')]))

def test_experimental_loader_rejects_frozen_protocol(tmp_path):
    from types import SimpleNamespace
    from scripts.evaluate_extended_gppo import install_model
    model=GPPOActorCritic(embed_dim=16,n_layers=1)
    saved=tmp_path/'old.pth'
    torch.save(dict(protocol_version='phase14-pretrain-v9',model=model.state_dict()),saved)
    adapter=SimpleNamespace(model=model,device=torch.device('cpu'))
    with pytest.raises(ValueError):install_model(adapter,saved)

def test_checkpoint_selection_never_rewards_wrong_priority_even_if_cost_is_large():
    from scripts.train_extended_gppo import selection_key
    correct={'priority_errors':0,'mean_observed_cost_regret':2000.}
    wrong={'priority_errors':1,'mean_observed_cost_regret':-100000.}
    assert selection_key(correct)<selection_key(wrong)

def test_true_latency_return_censors_missing_target_and_excludes_past_delay():
    from scripts.train_extended_gppo_online import latency_return
    assert latency_return([2.,10.],0.,10.)==pytest.approx(-.6)
    assert latency_return([2.,10.],3.,10.)==pytest.approx(-.35)

def test_ppo_clip_does_not_reward_unbounded_probability_increase():
    from scripts.train_extended_gppo_online import clipped_policy_loss
    loss=clipped_policy_loss(torch.log(torch.tensor([4.])),torch.tensor([0.]),torch.tensor([1.]),.2)
    assert loss.item()==pytest.approx(-1.2)

def test_heading_augmentation_uses_delayed_public_heading_and_id_mapping():
    import numpy as np
    from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
    from scripts.extended_gppo_heading import HeadingObservations,heading_batch
    tracker=HeadingObservations(age_fraction=.5)
    tracker.observe({9:{'heading':350.},2:{'heading':180.}})
    tracker.observe({9:{'heading':10.},2:{'heading':180.}})
    graph=TaskGraph([UAVState(2,position=(0.,0.)),UAVState(9,position=(0.,0.))],
        [Subtask(4,TaskType.TARGET_SEARCH,2,position=(10.,0.))])
    batch=heading_batch(graph,tracker,'cpu')
    assert batch['edge_features'].shape==(1,1,2,6)
    assert batch['edge_features'][0,0,0,3:].tolist()==pytest.approx([-1.,0.,1.],abs=1e-6)
    assert batch['edge_features'][0,0,1,3:].tolist()==pytest.approx([1.,0.,1.],abs=1e-6)

def test_missing_public_heading_is_marked_unknown_not_read_from_runtime():
    from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
    from scripts.extended_gppo_heading import HeadingObservations,heading_batch
    tracker=HeadingObservations();tracker.observe({9:{'heading':0.}});tracker.observe({})
    graph=TaskGraph([UAVState(9,position=(0.,0.))],[Subtask(4,TaskType.TARGET_SEARCH,2,position=(10.,0.))])
    assert heading_batch(graph,tracker,'cpu')['edge_features'][0,0,0,3:].tolist()==[0.,0.,0.]

def test_heading_model_initialization_preserves_original_policy_exactly():
    from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
    from integrations.gppo.gppo_model import batch_single_task_graph
    from scripts.extended_gppo_heading import HeadingObservations,heading_batch,initialize_heading_model
    torch.manual_seed(42)
    old=GPPOActorCritic();old.eval()
    graph=TaskGraph([UAVState(9,position=(0.,0.)),UAVState(2,position=(1.,1.))],
        [Subtask(4,TaskType.TARGET_SEARCH,2,position=(10.,0.))])
    new=initialize_heading_model(old.state_dict(),'cpu');new.eval()
    tracker=HeadingObservations();tracker.observe({9:{'heading':0.},2:{'heading':90.}})
    with torch.no_grad():
        before=old(**batch_single_task_graph(graph,'cpu'));after=new(**heading_batch(graph,tracker,'cpu'))
    assert torch.allclose(before['flat_logits'],after['flat_logits'],atol=1e-6,rtol=1e-6)
    assert torch.allclose(before['value'],after['value'],atol=1e-6,rtol=1e-6)

def test_actor_input_snapshot_preserves_extra_features_and_is_independent():
    import json
    from scripts.gppo_training_records import actor_input_record
    batch={'edge_features':torch.tensor([[[[1.,2.,3.,.5,.6,1.]]]]),
           'action_mask':torch.tensor([[[False]]])}
    saved=actor_input_record(batch,time=1.,action=0,uav_ids=[9],task_ids=[4])
    batch['edge_features'].zero_()
    assert saved['batch']['edge_features'][0][0][0]==pytest.approx([1.,2.,3.,.5,.6,1.])
    assert saved['chosen']=={'uav_id':9,'subtask_id':4}
    json.dumps(saved)


def test_reward_scale_reset_preserves_actor_and_encoder():
    from scripts.train_extended_gppo_heading import reset_value_head
    model=GPPOActorCritic(embed_dim=16,n_layers=1,edge_feature_dim=6)
    before=copy.deepcopy(model.state_dict())
    reset_value_head(model)
    after=model.state_dict()
    output_params={id(p) for p in model.critic.value[-1].parameters()}
    output_names={name for name,p in model.named_parameters() if id(p) in output_params}
    assert len(output_names)==2
    for name,tensor in after.items():
        if name in output_names:
            assert torch.count_nonzero(tensor)==0
        else:
            assert torch.equal(tensor,before[name])
    assert torch.count_nonzero(model.critic.value[-1].weight)==0
    assert torch.count_nonzero(model.critic.value[-1].bias)==0
