"""On-policy episode accumulation; no optimizer or environment operations."""
import hashlib
import math
import torch

def model_state_fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

class FrozenBehaviorBatch:
    def __init__(self,capacity):
        if not isinstance(capacity,int) or capacity<1:raise ValueError('Positive integer episode capacity required')
        self.capacity=capacity
        self.clear()

    def clear(self):
        self.transitions=[]
        self.episodes=0
        self.policy_sha256=None

    def begin(self,model):
        if self.episodes or self.policy_sha256 is not None:
            raise RuntimeError('Clear before starting another behavior policy')
        self.policy_sha256=model_state_fingerprint(model)

    @property
    def ready(self):
        return self.episodes>=self.capacity

    def add_episode(self,model,transitions,returns):
        if self.ready:raise RuntimeError('Update and clear the completed rollout batch first')
        if not transitions or len(transitions)!=len(returns) or not all(math.isfinite(float(r)) for r in returns):
            raise ValueError('Finite aligned rewards and nonempty transitions required')
        fingerprint=model_state_fingerprint(model)
        if self.policy_sha256 is not None and fingerprint!=self.policy_sha256:
            raise RuntimeError('Behavior model changed while collecting a rollout batch')
        self.policy_sha256=fingerprint
        self.transitions.extend({**tr,'return':float(reward)} for tr,reward in zip(transitions,returns))
        self.episodes+=1

def validate_reward_horizon(checkpoint,horizon,reset_critic):
    if horizon<=0:raise ValueError('Positive reward horizon required')
    if checkpoint.get('training_stage')!='online PPO' or reset_critic:return
    previous=checkpoint.get('training_args',{}).get('steps')
    if previous is None or previous!=horizon:
        raise ValueError('Online reward horizon changed or missing; explicitly reset critic before continuing')
