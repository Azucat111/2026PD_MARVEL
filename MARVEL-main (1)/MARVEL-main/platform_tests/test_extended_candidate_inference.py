import copy
from pathlib import Path
import hashlib
import pytest
import torch
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.gppo_model import GPPOActorCritic
from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
from scripts.extended_gppo_heading import HeadingObservations,heading_batch,HEADING_PROTOCOL
BASE=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
pytestmark=pytest.mark.skipif(not BASE.exists(),reason='Original frozen checkpoint unavailable')

def candidate(tmp_path,protocol=HEADING_PROTOCOL):
 torch.manual_seed(213);torch.set_num_threads(1)
 model=GPPOActorCritic(edge_feature_dim=6);file=tmp_path/'candidate.pth'
 torch.save(dict(protocol_version=protocol,model=model.state_dict()),file)
 return file,model

def test_extended_public_heading_features_logits_and_argmax_match_reference(tmp_path):
 file,reference=candidate(tmp_path);reference.eval();adapter=ExtendedGPPOInferenceAdapter(BASE,file,device='cpu')
 tracker=HeadingObservations(age_fraction=.05)
 for obs in [{2:{'heading':350},9:{'heading':180}},{2:{'heading':10},9:{'heading':120}}]:
  adapter.observe(obs);tracker.observe(obs)
 graph=TaskGraph([UAVState(9,position=(4,2)),UAVState(2,position=(0,0))],[Subtask(7,TaskType.TARGET_SEARCH,2,position=(10,0)),Subtask(3,TaskType.RELAY,2,position=(1,3))])
 graph.action_mask[0,0]=True
 batch=heading_batch(graph,tracker,'cpu')
 with torch.no_grad():expected=reference(**batch)
 choice=adapter.assign(graph)
 for key,value in batch.items():assert torch.equal(value,adapter.last_batch[key])
 assert choice.flat_action==int(expected['flat_logits'][0].argmax())
 assert choice.uav_id==graph.uav_states[choice.uav_index].uav_id
 assert choice.subtask_id==graph.subtasks[choice.task_index].subtask_id
 assert choice.value==pytest.approx(float(expected['value'][0]),abs=1e-7)
 assert adapter.model_protocol==HEADING_PROTOCOL
 assert adapter.checkpoint['protocol_version']=='phase14-pretrain-v9'

def test_candidate_wrong_protocol_is_rejected_without_weakening_frozen_loader(tmp_path):
 file,_=candidate(tmp_path,protocol='phase14-pretrain-v9')
 with pytest.raises(ValueError):ExtendedGPPOInferenceAdapter(BASE,file)
 file,_=candidate(tmp_path)
 with pytest.raises(RuntimeError):load_frozen_gppo(file)

def test_expected_candidate_hash_mismatch_fails_closed(tmp_path):
 file,_=candidate(tmp_path)
 with pytest.raises(ValueError):ExtendedGPPOInferenceAdapter(BASE,file,expected_sha256='0'*64)
 adapter=ExtendedGPPOInferenceAdapter(BASE,file,expected_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 assert adapter.candidate_sha256==hashlib.sha256(file.read_bytes()).hexdigest()

def test_all_masked_and_missing_public_heading_preserve_validity(tmp_path):
 file,_=candidate(tmp_path);adapter=ExtendedGPPOInferenceAdapter(BASE,file)
 graph=TaskGraph([UAVState(9,position=(0,0))],[Subtask(7,TaskType.TARGET_SEARCH,2,position=(10,0))])
 adapter.observe({});adapter.assign(graph)
 assert adapter.last_batch['edge_features'][0,0,0,3:].tolist()==[0.,0.,0.]
 graph.action_mask[:]=True
 assert adapter.assign(graph) is None

def test_loading_and_inference_do_not_mutate_the_original_frozen_checkpoint(tmp_path):
 before=hashlib.sha256(BASE.read_bytes()).hexdigest();file,_=candidate(tmp_path)
 adapter=ExtendedGPPOInferenceAdapter(BASE,file)
 adapter.observe({9:{'heading':0}})
 adapter.assign(TaskGraph([UAVState(9,position=(0,0))],[Subtask(7,TaskType.TARGET_SEARCH,2,position=(10,0))]))
 assert hashlib.sha256(BASE.read_bytes()).hexdigest()==before

def test_native_rejects_explicit_experimental_candidate_before_loading_it():
 from types import SimpleNamespace
 from integrations.gppo.event_allocator import GPPOEventAllocator
 with pytest.raises(ValueError,match='extended'):
  GPPOEventAllocator(SimpleNamespace(geometry_mode='marvel_native'),str(BASE),extended_checkpoint='/not/a/candidate.pth')

def test_candidate_heading_snapshot_clock_ignores_extra_physics_calls():
 from types import SimpleNamespace
 from integrations.gppo.scheduler import GPPOTaskScheduler
 calls=[];sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
 sched.runtime=SimpleNamespace(current_step=0)
 sched.profile=SimpleNamespace(high_level_interval_steps=10)
 sched.allocator=SimpleNamespace(adapter=SimpleNamespace(observe=lambda obs:calls.append(obs[2]['heading'])))
 sched._search_recovery=None
 for step,heading in [(0,0),(1,45),(9,90),(10,120)]:
  sched.runtime.current_step=step;sched.observe({2:{'heading':heading}})
 assert calls==[0,120]
