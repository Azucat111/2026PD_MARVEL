"""Prelaunch actor-preserving value-output/optimizer reset verification; no training."""
from pathlib import Path
import sys,json,hashlib,torch,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from integrations.gppo.gppo_model import GPPOActorCritic
from train_extended_gppo_batched import reset_value_head
from gppo_rollout_batch import validate_reward_horizon
path=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt17/checkpoint_batch_015.pth';digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest=='1cda34ecb47ec1eb411e3bf47f30351bbec1a7d48b96ff6297f70ac0739cf22a'
cp=torch.load(path,map_location='cpu',weights_only=False);model=GPPOActorCritic(edge_feature_dim=6);model.load_state_dict(cp['model']);before={k:v.clone() for k,v in model.state_dict().items()};reset_value_head(model);optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
after=model.state_dict();actor_keys=[k for k in before if k.startswith(('actor.','ahgnn.'))];assert all(torch.equal(before[k],after[k]) for k in actor_keys);assert not optimizer.state
assert torch.count_nonzero(model.critic.value[-1].weight)==0 and torch.count_nonzero(model.critic.value[-1].bias)==0
rejected=False
try:validate_reward_horizon(cp,150,False)
except ValueError:rejected=True
assert rejected;validate_reward_horizon(cp,150,True)
source=(ROOT/'scripts/train_extended_gppo_parallel_role_normalized_attempt18.py').read_text();assert "reset_value_head(model)" in source and "optimizer.load_state_dict" not in source and "args.steps=150;args.reset_critic=True" in source;ast.parse(source)
result=dict(passed=True,training_performed=False,old_horizon=cp['training_args']['steps'],new_horizon=150,actor_and_shared_encoder_exactly_preserved=True,critic_final_output_zero=True,optimizer_fresh=True,unreset_horizon_transition_rejected=True,initial_checkpoint_sha256=digest,coordinator_sha256=hashlib.sha256(source.encode()).hexdigest(),acceptance_claim=False)
(ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt2/prelaunch_H150_actor_reset_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
