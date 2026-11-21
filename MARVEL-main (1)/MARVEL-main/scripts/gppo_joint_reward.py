"""Training-only suffix rewards from actual runtime radio outcomes, never actor input."""
import math

def validate_reward_protocol(checkpoint,joint,reset_critic,relay_weight=.25):
    requested='joint_physical_v1' if joint else 'search_latency_v1'
    existing=checkpoint.get('reward_protocol','search_latency_v1')
    weight_changed=joint and existing==requested and checkpoint.get('training_args',{}).get('relay_reward_weight',.25)!=relay_weight
    if (existing!=requested or weight_changed) and not reset_critic:
        raise ValueError('Reward protocol changed: reset critic and optimizer explicitly')

def remaining_connectivity_deficit(samples,decision_time,horizon,goal=.9):
    if not all(math.isfinite(x) for x in (decision_time,horizon,goal)) or horizon<=0 or not 0<=decision_time<=horizon or not 0<goal<=1:
        raise ValueError('Invalid physical reward interval')
    if not samples:raise ValueError('No actual physical radio trace')
    previous=0.;total=0.
    for time,ratio in samples:
        if not math.isfinite(time) or time<=previous or not math.isfinite(ratio) or not 0<=ratio<=1:
            raise ValueError('Invalid ordered physical radio trace')
        # Right-endpoint physical tick outcome over (previous,time].
        duration=max(0.,min(time,horizon)-max(previous,decision_time))
        total+=duration*max(0.,goal-ratio)/goal
        previous=time
    if previous<horizon-1e-7:raise ValueError('Incomplete physical radio horizon')
    return total/horizon
