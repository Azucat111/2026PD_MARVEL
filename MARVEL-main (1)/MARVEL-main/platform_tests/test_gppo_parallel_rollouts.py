import importlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]

def api():
 assert importlib.util.find_spec('scripts.gppo_parallel_rollouts') is not None, 'Parallel rollout execution is not implemented'
 return importlib.import_module('scripts.gppo_parallel_rollouts')

def test_global_domain_cycle_survives_worker_process_indices():
 m=api()
 assert [m.episode_domain(i) for i in range(8,12)]==[(30,False),(60,False),(30,True),(60,True)]

def test_mixed_behavior_policies_never_form_update_batch():
 m=api()
 with pytest.raises(ValueError,match='behavior'):
  m.validate_frozen_rows([{'behavior_sha256':'a'},{'behavior_sha256':'b'}],'a')

def test_actual_saved_episode_restores_public_teacher_and_roles():
 m=api();folder=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt11'
 row,transitions=m.load_captured_episode(folder,0,'cpu')
 assert len(transitions)==5
 assert transitions[0]['action']==23 and transitions[0]['role']==1
 assert all(t['return']==r for t,r in zip(transitions,row['returns']))
 assert all(t['batch']['action_mask'].dtype==torch.bool for t in transitions)

def test_real_saved_eight_episode_update_matches_checkpoint():
 m=api();folder=ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt11'
 result=m.replay_historical_update(folder,ROOT/'artifacts/extended_gppo_joint_batch_20261011_attempt10/checkpoint_batch_000.pth','cuda')
 assert result['max_model_error']<1e-6
 assert result['max_optimizer_error']<1e-6
 assert result['transitions']==44

def test_rollout_compiler_rejects_missing_collection_boundary():
 m=api()
 with pytest.raises(ValueError,match='boundary'):
  m.compile_rollout_source('def main(): pass',8)
