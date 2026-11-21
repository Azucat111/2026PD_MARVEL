import pytest
import torch
from scripts.gppo_rollout_batch import FrozenBehaviorBatch

def test_independent_episodes_share_one_immutable_behavior_policy():
 model=torch.nn.Linear(1,1);batch=FrozenBehaviorBatch(2)
 batch.add_episode(model,[{'action':0}],[.1])
 assert not batch.ready and batch.episodes==1
 batch.add_episode(model,[{'action':1},{'action':2}],[.2,.3])
 assert batch.ready and batch.episodes==2
 assert [t['return'] for t in batch.transitions]==[.1,.2,.3]
 batch.clear()
 assert not batch.ready and not batch.transitions

def test_mid_batch_policy_updates_are_rejected():
 model=torch.nn.Linear(1,1);batch=FrozenBehaviorBatch(2)
 batch.add_episode(model,[{'action':0}],[.1])
 with torch.no_grad():model.weight.add_(1)
 with pytest.raises(RuntimeError):batch.add_episode(model,[{'action':1}],[.2])
 assert batch.episodes==1
 batch.clear();batch.add_episode(model,[{'action':1}],[.2])
 assert batch.episodes==1

def test_invalid_reward_alignment_never_enters_rollout_batch():
 batch=FrozenBehaviorBatch(2);model=torch.nn.Linear(1,1)
 with pytest.raises(ValueError):batch.add_episode(model,[{'action':0}],[])
 with pytest.raises(ValueError):batch.add_episode(model,[{'action':0}],[float('nan')])
 assert batch.episodes==0

def test_policy_is_frozen_before_first_rollout_not_after_it():
 model=torch.nn.Linear(1,1);batch=FrozenBehaviorBatch(2);batch.begin(model)
 with torch.no_grad():model.bias.add_(1)
 with pytest.raises(RuntimeError):batch.add_episode(model,[{'action':0}],[.1])
 assert batch.episodes==0

def test_reward_scale_change_requires_explicit_critic_reset():
 from scripts.gppo_rollout_batch import validate_reward_horizon
 cp={'training_stage':'online PPO','training_args':{'steps':80}}
 validate_reward_horizon(cp,80,False)
 with pytest.raises(ValueError):validate_reward_horizon(cp,40,False)
 validate_reward_horizon(cp,40,True)
 with pytest.raises(ValueError):validate_reward_horizon({'training_stage':'online PPO'},80,False)
