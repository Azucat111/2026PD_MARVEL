"""Paired-return model and admitted-index safety contracts."""
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_joint_public_preference import validated_score
from integrations.gppo.public_pair_preference import (
    PublicPairPreference, event_features, make_checkpoint, search_donor_regret,
)
from integrations.gppo.task_graph import Subtask, TaskGraph, TaskType, UAVState
from run_joint_autoresearch import action_at


def event():
    return dict(stale_xy={0:[0,0], 1:[5,0], 2:[7,0], 3:[50,0]}, time_seconds=3.,
                search_before={}, relay_before={}, unavailable=[], range_m=20.,
                rows=[dict(flat=1, slot=10, uav=0, role=1, p=[0,0], anchor=[40,0],
                           waypoint=[15,0], logit=2., total=45.),
                      dict(flat=4, slot=11, uav=1, role=1, p=[5,0], anchor=[8,0],
                           waypoint=[8,0], logit=0., total=8.),
                      dict(flat=8, slot=12, uav=2, role=2, p=[7,0], anchor=[29,0],
                           waypoint=[16,0], logit=-1., total=23.)])


def fitted(raw, positive, mode):
    context, features = event_features(raw)
    row = dict(control_seconds=81., treatment_seconds=36., context=context.tolist(),
               old_features=features[0].tolist(), new_features=features[positive].tolist(),
               old_logit=raw["rows"][0]["logit"], new_logit=raw["rows"][positive]["logit"])
    return PublicPairPreference(make_checkpoint([row], base_sha="test", mode=mode,
                                                context_scale=1., action_scale=.85, support=.42))


def test_full_return_contrast_reorders_legal_candidates():
    raw = event()
    index, evidence = fitted(raw, 1, "within_role").choose(raw)
    assert index == 4
    assert evidence["legal_indices"] == [1,4,8]
    assert all(prob>0 for prob in evidence["probs"])
    assert sum(evidence["probs"]) == pytest.approx(1.)


def test_cross_role_data_can_change_role_and_within_role_cannot():
    raw = event()
    assert fitted(raw, 2, "joint").choose(raw)[0] == 8
    index, evidence = fitted(raw, 2, "within_role").choose(raw)
    assert index in (1,4)
    assert 8 in evidence["legal_indices"]


def test_far_state_falls_back_and_features_exclude_identifiers():
    raw = event()
    changed = deepcopy(raw)
    changed["rows"][0]["slot"] = 98
    np.testing.assert_allclose(event_features(raw)[1], event_features(changed)[1])
    changed["time_seconds"] = 30.
    index, evidence = fitted(raw, 1, "joint").choose(changed)
    assert index == 1
    np.testing.assert_array_equal(evidence["logits"], evidence["prior_logits"])


def test_selector_cannot_override_hard_mask():
    graph = TaskGraph([UAVState(10),UAVState(11)],
                      [Subtask(1,TaskType.TARGET_SEARCH,1.,position=(0.,0.))])
    graph.action_mask[0,1] = True
    assert action_at(graph,0,0.).uav_id == 10
    with pytest.raises(ValueError,match="illegal"):
        action_at(graph,1,0.)


def test_fit_rejects_partial_physical_returns():
    with pytest.raises(ValueError,match="H300"):
        validated_score(dict(trace=[{}],commitments=[[]],score=None))


def test_checkpoint_rejects_bad_digest(tmp_path):
    path = tmp_path/"checkpoint.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="hash"):
        PublicPairPreference.load(path,"0"*64)


def test_donor_regret_exposes_shared_search_resource_using_only_legal_rows():
    raw = event()
    raw["rows"][2]["uav"] = 0
    assert search_donor_regret(raw["rows"],0) == pytest.approx(2.)
    assert search_donor_regret(raw["rows"],1) == 0.
    _, features = event_features(raw,opportunity=True)
    assert features.shape == (3,15)
    assert features[:, -1].tolist() == [0.,0.,.5]
