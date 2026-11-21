import pytest
from scripts.gppo_joint_reward import remaining_connectivity_deficit,validate_reward_protocol

def test_bridge_reduces_actual_suffix_deficit():
    assert remaining_connectivity_deficit([(10,.45),(20,.45),(30,.9),(40,.9)],0,40)==pytest.approx(.25)
    assert remaining_connectivity_deficit([(10,.45),(20,.45),(30,.9),(40,.9)],15,40)==pytest.approx(.0625)
    assert remaining_connectivity_deficit([(10,.45),(20,.45),(30,.45),(40,.45)],0,40)==pytest.approx(.5)

def test_goal_connectivity_has_no_deficit():
    assert remaining_connectivity_deficit([(1,.9),(2,1.)],0,2)==0
    assert remaining_connectivity_deficit([(1,.45),(2,.45)],2,2)==0

@pytest.mark.parametrize('samples',[[],[(1,.5)],[(2,.5),(1,.5)],[(2,float('nan'))],[(2,1.1)]])
def test_incomplete_or_invalid_physical_trace_rejected(samples):
    with pytest.raises(ValueError):remaining_connectivity_deficit(samples,0,2)

def test_joint_reward_requires_critic_reset_for_old_search_checkpoint():
    cp={'training_stage':'online PPO'}
    with pytest.raises(ValueError):validate_reward_protocol(cp,True,False)
    validate_reward_protocol(cp,True,True)
    validate_reward_protocol({'reward_protocol':'joint_physical_v1'},True,False)
    with pytest.raises(ValueError):validate_reward_protocol({'reward_protocol':'joint_physical_v1'},False,False)

def test_changed_joint_reward_weight_requires_reset():
    cp={'reward_protocol':'joint_physical_v1','training_args':{'relay_reward_weight':.25}}
    with pytest.raises(ValueError):validate_reward_protocol(cp,True,False,1.)
    validate_reward_protocol(cp,True,True,1.)
