from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_event_ppo import action_distribution, assign_returns, guarded_epoch, verify_replay
from gppo_rollout_batch import model_state_fingerprint


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0., 20., 1., 3.]))
        self.value = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, action_mask):
        return {"flat_logits": self.logits[:action_mask.numel()].unsqueeze(0),
                "value": self.value.unsqueeze(0)}


def transition(model, size, *, selected=0):
    batch = {"action_mask": torch.tensor([[[False, True, False, False][:size]]])}
    with torch.no_grad():
        dist, value, legal = action_distribution(model, batch, 3.)
    flat = int(legal.nonzero()[selected])
    return dict(batch=batch, selected_flat=flat, selected_legal_index=selected,
                task_ids=[101], uav_ids=list(range(size)), roles=[1], old_value=float(value),
                old_logp=float(dist.logits[selected]), old_log_probs=dist.logits.clone(),
                receipt=dict(assignment=dict(slot_id=101, uav_id=flat, task_type=1)),
                return_target=-.5, advantage=.3)


def test_model_replay_with_different_shapes_masks_and_tampered_mapping():
    model = TinyPolicy().eval()
    for size in (3, 4):
        row = transition(model, size, selected=1)
        assert row["selected_flat"] == 2
        verify_replay(model, row, 3.)
        row["selected_flat"] = 1
        with pytest.raises(ValueError, match="mask"):
            verify_replay(model, row, 3.)


def test_elapsed_cost_with_missing_decision_intervals_and_completion_cutoff():
    events = [dict(step=s, old_value=.1) for s in (10, 40, 80)]
    prefix = assign_returns(events, end_step=3000, dt=.1, cost_seconds=6.2)
    assert math.isclose(prefix+sum(r["reward"] for r in events), -6.2/300.)
    assert [r["duration_seconds"] for r in events] == [3., 4., 292.]
    assert events[-1]["reward"] == 0 and events[-1]["terminal"]
    assert events[0]["advantage"] == pytest.approx(-5.2/300.-.1)


def test_terminal_penalty_for_failure_and_separate_real_bootstrap():
    full = [dict(step=10, old_value=0.)]
    prefix = assign_returns(full, end_step=3000, dt=.1, cost_seconds=300.)
    assert prefix + full[0]["return_target"] == pytest.approx(-1.)
    prefix_row = [dict(step=10, old_value=0.)]
    assign_returns(prefix_row, end_step=20, dt=.1, cost_seconds=2.,
                   bootstrap=-.2, terminal=False)
    assert prefix_row[0]["return_target"] == pytest.approx(-.2-1/300.)
    assert prefix_row[0]["truncated"] and not prefix_row[0]["terminal"]


def test_kl_rejection_restores_weights_and_populated_adam():
    model = TinyPolicy().eval()
    adam = torch.optim.Adam(model.parameters(), lr=.03)
    assert guarded_epoch(model, adam, [transition(model, 4)], 3.)["accepted"]
    before = deepcopy(adam.state_dict())
    fingerprint = model_state_fingerprint(model)
    adam.param_groups[0]["lr"] = 10.
    before["param_groups"][0]["lr"] = 10.
    row = transition(model, 3)
    result = guarded_epoch(model, adam, [row], 3.)
    assert not result["accepted"] and result["max_state_exact_kl"] > .05
    assert model_state_fingerprint(model) == fingerprint
    after = adam.state_dict()
    assert after["param_groups"] == before["param_groups"]
    for key in before["state"]:
        for name in before["state"][key]:
            torch.testing.assert_close(after["state"][key][name], before["state"][key][name], atol=0, rtol=0)


def test_pair_reducer_refuses_unequal_initial_state_and_no_score_for_truncation():
    from summarize_joint_ppo_evaluation import paired_summary

    first = dict(case="x", arm="R1", trace=[], status="truncated", commitments=[], offers=[],
                 initial_sha256="a", config_sha256="c", checkpoint_sha256="original",
                 compatibility_policy=False, seconds=1.)
    second = dict(first, arm="P1", checkpoint_sha256="p1")
    assert paired_summary([first, second], ["x"], p1_sha256="p1")[
        "pooled_comparable_pairs"]["R1"] == {"count": 0, "cases": [], "percent": None}
    with pytest.raises(ValueError, match="pairing"):
        paired_summary([first, dict(second, initial_sha256="b")], ["x"], p1_sha256="p1")
    with pytest.raises(ValueError, match="Truncated"):
        paired_summary([first, dict(second, score={})], ["x"], p1_sha256="p1")


def test_full_four_arm_contrasts_count_complete_failure():
    from gppo_joint_completion_metric import joint_completion_metric
    from summarize_joint_ppo_evaluation import paired_summary

    def run(arm, ready_at):
        trace = [dict(time=t*.1, connectivity=1., nonrelay_complete=t >= ready_at,
                      safety_active=[], escapes=0, safety_records=0, collisions=0,
                      no_fly_violations=0) for t in range(1, 3001)]
        return dict(case="x", arm=arm, trace=trace, status="complete", commitments=[], offers=[],
                    initial_sha256="same", config_sha256="same", checkpoint_sha256=arm,
                    compatibility_policy=False, seconds=1., score=joint_completion_metric(trace, 300))

    result = paired_summary([run("R0", 30), run("P0", 3001), run("R1", 20), run("P1", 10)],
                            ["x"], p1_sha256="P1")
    contrasts = result["pooled_comparable_pairs"]
    assert contrasts["R0"]["percent"] == pytest.approx(100*(1-4/6))
    assert contrasts["R1/R0"]["percent"] == pytest.approx(100*(1-5/6))
    assert contrasts["P0/R0"]["percent"] == pytest.approx(100*(1-300/6))
    assert contrasts["P0/R0"]["count"] == 1
    assert not result["case_arms"]["x"]["P0"]["score"]["joint_success"]
