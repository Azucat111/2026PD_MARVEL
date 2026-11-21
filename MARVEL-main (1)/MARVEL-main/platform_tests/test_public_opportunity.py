import numpy as np
import torch
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
from integrations.gppo.heading_features import HeadingObservations,heading_batch
from integrations.gppo.opportunity_features import opportunity_batch,initialize_opportunity_model
from integrations.gppo.gppo_model import GPPOActorCritic

def graph(kind=TaskType.TARGET_SEARCH):
 return TaskGraph([UAVState(0,position=(0.,0.),velocity=2.),UAVState(1,position=(20.,0.),velocity=1.)],[Subtask(100000,kind,1.,position=(14.,0.))])

def observer():
 o=HeadingObservations();o.observe({0:{'heading':0},1:{'heading':180}})
 o.public_search_context={'pending_priors':[[0,[14.,0.]]],'reserved_uids':[],'sensor_range':10.}
 return o

def test_sensor_eta_is_absolute_and_masks_unchanged():
 g=graph();o=observer();b=opportunity_batch(g,o,'cpu');old=heading_batch(g,o,'cpu')
 assert b['edge_features'].shape[-1]==9
 torch.testing.assert_close(b['edge_features'][...,:6],old['edge_features'],rtol=0,atol=0)
 torch.testing.assert_close(b['action_mask'],old['action_mask'])
 assert float(b['edge_features'][0,0,0,6])==np.float32(2./80)
 assert float(b['edge_features'][0,0,1,6])==0.

def test_relay_sees_pending_search_opportunity():
 g=graph(TaskType.RELAY);o=observer();o.public_search_context['pending_priors']=[[0,[0.,0.]]]
 b=opportunity_batch(g,o,'cpu')['edge_features']
 assert float(b[0,0,0,7])==np.float32(10./80)
 assert float(b[0,0,1,7])==0.
 assert torch.all(b[...,8]==1)
 assert torch.all(b[...,6]==0)

def test_reserved_uav_not_counted_as_alternative():
 g=graph(TaskType.RELAY);o=observer();o.public_search_context['pending_priors']=[[0,[0.,0.]]];o.public_search_context['reserved_uids']=[1]
 b=opportunity_batch(g,o,'cpu')['edge_features'];assert float(b[0,0,0,7])==1.

def test_missing_context_not_invented():
 g=graph();o=HeadingObservations();b=opportunity_batch(g,o,'cpu')['edge_features'];assert torch.all(b[...,7:]==0)

def test_zero_speed_and_nonfinite_prior():
 g=graph();g.uav_states[0].velocity=0.;o=observer();b=opportunity_batch(g,o,'cpu')['edge_features'];assert torch.isfinite(b).all();assert float(b[0,0,0,6])==1.
 o.public_search_context['pending_priors']=[[0,[float('nan'),0.]]]
 import pytest
 with pytest.raises(ValueError):opportunity_batch(g,o,'cpu')

def test_zero_column_warmstart_preserves_actor_and_critic():
 torch.manual_seed(7);old=GPPOActorCritic(edge_feature_dim=6).eval();new=initialize_opportunity_model(old.state_dict(),'cpu');g=graph();o=observer()
 with torch.no_grad():a=old(**heading_batch(g,o,'cpu'));b=new(**opportunity_batch(g,o,'cpu'))
 for key in ['flat_logits','value']:
  torch.testing.assert_close(a[key],b[key],rtol=0,atol=1e-6)
 for key,value in old.state_dict().items():
  if key in ['ahgnn.edge_input.weight','actor.edge_proj.weight']:
   torch.testing.assert_close(new.state_dict()[key][:,:6],value,rtol=0,atol=0);assert torch.all(new.state_dict()[key][:,6:]==0)
  else:torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)

def test_context_only_public_projection_and_tuple_relay_assignments():
 from types import SimpleNamespace as N
 from integrations.gppo.opportunity_features import public_search_context
 scheduler=N(heat_by_id={0:N(position=(1.,2.)),1:N(position=(3.,4.)),2:N(position=(5.,6.))},serviced_heat_ids={2},search_assignments={1:8},relay_assignments={100:(9,np.array([40.,40.]))},safety=N(active_uav_ids={7}),runtime=N(sensor=N(sensor_range=12.)))
 assert public_search_context(scheduler)==dict(pending_priors=[[0,[1.,2.]]],reserved_uids=[7,8,9],sensor_range=12.)

def test_production_opportunity_loader_requires_public_provider(tmp_path):
 from pathlib import Path
 from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
 from integrations.gppo.opportunity_features import OPPORTUNITY_PROTOCOL
 import pytest
 model=GPPOActorCritic(edge_feature_dim=9);file=tmp_path/'opportunity.pth'
 torch.save(dict(protocol_version=OPPORTUNITY_PROTOCOL,model=model.state_dict()),file)
 adapter=ExtendedGPPOInferenceAdapter(Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth'),file)
 with pytest.raises(RuntimeError,match='context'):adapter.assign(graph())
 adapter.public_search_context_provider=lambda:observer().public_search_context
 adapter.observe({0:{'heading':0},1:{'heading':180}})
 choice=adapter.assign(graph());assert choice is not None
 torch.testing.assert_close(adapter.last_batch['edge_features'],opportunity_batch(graph(),observer(),'cpu')['edge_features'])

def test_saved_public_snapshot_reconstructs_all_nine_features_and_rejects_tamper():
 import sys,copy
 from pathlib import Path
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from audit_opportunity_inputs import audit_record,capture_public_snapshot
 g=graph();o=observer();batch=opportunity_batch(g,o,'cpu')
 rec=dict(time=0.,uav_ids=[0,1],task_ids=[100000],batch={k:v.tolist() for k,v in batch.items()},public_heading_control_snapshot=dict(interval_steps=10,physics_dt=.1,control_index=0,age_fraction=.05,current=o.current,previous=o.previous,public_uav_positions=[list(u.position) for u in g.uav_states],public_task_positions=[list(t.position) for t in g.subtasks]),public_opportunity_snapshot=capture_public_snapshot(g,o.public_search_context))
 assert audit_record(rec)<1e-6
 import pytest
 corrupt=copy.deepcopy(rec);corrupt['batch']['edge_features'][0][0][0][7]=.5
 with pytest.raises(AssertionError):audit_record(corrupt)

def test_isolated_training_extension_compiles_without_replacing_physics():
 import sys,ast
 from pathlib import Path
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from opportunity_training_source import training_source
 from gppo_parallel_rollouts import compile_rollout_source
 source=training_source();compile_rollout_source(source,0)
 assert 'public_search_context(self.scheduler)' in source
 assert 'public_opportunity_snapshot' in source
 assert 'edge_feature_dim=9' in source
