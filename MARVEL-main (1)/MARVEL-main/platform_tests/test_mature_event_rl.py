"""SMDP timing, legal masking, actual receipts, and the two learning objectives."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from integrations.gppo.event_rl import (
    Learner, Settings, bellman, dqn_continuation, masked_argmax, masked_probabilities,
    n_step_episode, padded,
)
from integrations.gppo.event_rl_replay import (
    Decision, Transition, joint_cost_trace, load_replay, save_replay, transitions_from_events,
)
from run_mature_rl import behavior_distribution


def decision(step=10, pair_count=2):
    return Decision(step, f"sha:{step}", None, ((10, 0, 1), (20, 1, 2))[:pair_count],
                    ((0., 1.), (1., 0.))[:pair_count])


def record():
    following = decision(40, 1)
    return Transition(decision(), 1, -.1, 3., following, False, False, "30_static_1", 1, "test",
                      "30_static_1", .25, (.75, .25))


def test_double_q_selects_online_and_evaluates_target_on_next_legal_graph():
    online = torch.tensor([[6., 8., 999.], [1., 2., 3.]])
    target = torch.tensor([[90., 4., 999.], [5., 999., 6.]])
    legal = torch.tensor([[True, True, False], [True, False, True]])
    assert dqn_continuation(online, target, legal).tolist() == [4., 6.]


def test_mask_removes_illegal_argmax_and_all_its_probability():
    values, legal = padded([decision(10, 2), decision(20, 1)])
    scores = torch.tensor([[.2, .4], [1., 100.]])
    assert values.shape == (2, 2, 2)
    assert masked_argmax(scores, legal).tolist() == [1, 0]
    probs, _ = masked_probabilities(scores, legal)
    assert probs[1].tolist() == [1., 0.]
    assert torch.allclose(probs.sum(-1), torch.ones(2))
    with pytest.raises(ValueError, match="empty"):
        masked_argmax(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_smdp_discount_uses_physical_duration_and_never_bootstraps_terminal():
    reward = torch.tensor([-1., -2.])
    delta = torch.tensor([2., 4.])
    next_value = torch.tensor([4., float("nan")])
    terminated = torch.tensor([False, True])
    censored = torch.tensor([False, False])
    result = bellman(reward, delta, next_value, terminated, censored, gamma_per_second=.5)
    assert result.tolist() == [0., -2.]
    with pytest.raises(ValueError, match="censored"):
        bellman(reward, delta, next_value, terminated, torch.tensor([True, False]))


@pytest.mark.parametrize("algo", ["ddqn", "sac"])
def test_target_sync_with_actual_gradient_update(algo):
    torch.manual_seed(2)
    learner = Learner(Settings(algo, 2, 2, hidden=8, sync_every=2))
    before = {k: v.clone() for k,v in learner.critic.state_dict().items()}
    transitions = [record(), replace(record(), state=decision(40, 1), action=0, reward=-2.,
                                     dt=296., next_state=None, done=True)]
    one = learner.update(transitions)
    assert one["bootstrap_rows"] == 1
    assert any(not torch.equal(v, before[k]) for k,v in learner.critic.state_dict().items())
    assert all(torch.equal(v, before[k]) for k,v in learner.target.state_dict().items())
    learner.update(transitions)
    assert all(torch.equal(v, learner.critic.state_dict()[k]) for k,v in learner.target.state_dict().items())


def test_replay_roundtrip_preserves_masks_sequences_and_behavior_probability(tmp_path):
    p = tmp_path / "replay.json.gz"
    first = record()
    last = replace(first, state=first.next_state, next_state=None, dt=296.,
                   done=True, action=0, behavior_probability=1., behavior_distribution=(1.,))
    payload = [("ep", [first, last])]
    summary = {"coverage": {"transitions": 2}}
    save_replay(p, payload, summary)
    assert load_replay(p) == (payload, summary)


def test_multistep_cumulative_physical_discount():
    first = record()
    last = replace(first, state=first.next_state, next_state=None, dt=2., reward=-4.,
                   done=True, action=0)
    rows = n_step_episode([first, last], 3, .5)
    assert rows[0].dt == 5.
    assert rows[0].reward == pytest.approx(-.6)
    assert rows[0].done and rows[0].next_state is None


def fixture(unsafe=False):
    trace = [dict(time=t * .1, connectivity=1., nonrelay_complete=t >= 25, safety_active=[],
                  safety_records=0, escapes=0, no_fly_violations=0,
                  collisions=int(unsafe and t == 185)) for t in range(1, 3001)]
    receipts = [
        dict(slot_id=10, uav_id=0, task_type=1, flat_action=0),
        dict(slot_id=20, uav_id=1, task_type=2, flat_action=3),
    ]
    events = []
    commitments = [[] for _ in range(300)]
    for step, receipt in zip((10, 40), receipts, strict=True):
        commitments[step//10] = [receipt]
        rows = [
            dict(flat=0, slot=10, uav=0, role=1, anchor=[9, 0], waypoint=[6, 0], total=10., logit=2.),
            dict(flat=3, slot=20, uav=1, role=2, anchor=[4, 0], waypoint=[5, 0], total=7., logit=1.),
        ]
        events.append(dict(step=step, selected=[receipt["slot_id"], receipt["uav_id"]], receipt=receipt,
                           rows=rows, snapshot_version=f"test:{step}", stale_xy={0: [0,0], 1: [5,0]},
                           range_m=20., time_seconds=step*.1, search_before={}, relay_before={}, unavailable=[],
                           task_ids=[10,20], uav_ids=[0,1], roles=[1,2], mask=[[False,True],[True,False]]))
    summary = dict(trace=trace, status="complete", case="30_static_1", commitments=commitments,
                   score=dict(joint_success=not unsafe, secondary_joint_completion_seconds=300. if unsafe else 5.5))
    return summary, events


def test_terminal_correction_counts_late_safety_failure_and_exact_time():
    for unsafe in (False, True):
        episode, rows = fixture(unsafe)
        transitions, prefix = transitions_from_events(episode, rows, policy_id="test", group="30_static_1")
        assert transitions[0].dt == 3.
        assert transitions[0].reward == pytest.approx(-3/30.)
        assert transitions[0].next_state == transitions[1].state
        assert transitions[1].dt == pytest.approx(296.)
        success, provisional, charged = joint_cost_trace(episode["trace"])
        assert (success, provisional) == (not unsafe, 5.5)
        assert prefix + sum(t.reward for t in transitions) == pytest.approx(-charged/30.)
        assert transitions[1].done


def test_receipt_mask_and_missing_behavior_data_are_not_inferred():
    episode, events = fixture()
    rows, _ = transitions_from_events(episode, events, policy_id="test", group="30_static_1")
    assert rows[0].behavior_probability is None and rows[0].executed == (10,0,1)
    events[0]["mask"][0][0] = True
    with pytest.raises(ValueError, match="legal mask"):
        transitions_from_events(episode, events, policy_id="test", group="30_static_1")


def test_role_balanced_behavior_is_normalized_and_includes_actual_relay():
    entries = [
        dict(role=1, flat=3, logit=10, total=4),
        dict(role=1, flat=5, logit=7, total=3),
        dict(role=2, flat=11, logit=0, total=5),
    ]
    distribution = behavior_distribution(entries, 3, epsilon=.1, mixed_epsilon=.8, top=2)
    np.testing.assert_allclose(distribution, [.4,.2,.4])
    assert distribution.sum() == pytest.approx(1.)
