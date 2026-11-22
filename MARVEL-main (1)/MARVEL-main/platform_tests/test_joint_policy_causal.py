"""Audit integrity and attribution tests for Phase 3.5."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_joint_policy_causal import (  # noqa: E402
    commitments, first_change, first_policy_fork, forecast_retarget, threshold_merge, window_times,
)
from joint_policy_causal_trace import probe_state  # noqa: E402


def test_probe_is_optional_before_the_first_control_tick():
    assert probe_state(object()) is None


def test_stable_commitments_ignore_only_critic_metadata():
    a = dict(slot_id=10001, uav_id=2, task_type=1, value=0)
    assert commitments([[a]]) == commitments([[dict(a, value=-.08)]])
    assert commitments([[a]]) != commitments([[dict(a, uav_id=3)]])


def test_policy_fork_requires_same_pre_action_state_and_input():
    common = dict(step=30, selected=[1, 2], snapshot_version="a", input_sha256="b",
                  mask=[[False, False]], candidates=[[1, 2], [1, 3]])
    first = {30: common}
    second = {30: dict(common, selected=[1, 3])}
    assert first_policy_fork(first, second, {"observed": {"step": 31}}) == 30
    assert first_policy_fork(first, second, {"observed": {"step": 30}}) is None
    second[30]["input_sha256"] = "different"
    assert first_policy_fork(first, second, {}) is None
    assert first_change({30: common, 40: common}, {30: common}, ("selected",)) == {}


def test_qualified_window_restarts_after_radio_and_safety_interruptions():
    trace = [dict(time=t*.1, nonrelay_complete=t>=20, connectivity=0.9 if t not in (12,38) else .8,
                  safety_active=[22] if 45<=t<53 else [], escapes=0, safety_records=0)
             for t in range(1, 100)]
    outcome = window_times(trace)
    assert outcome["radio_first_09"] == pytest.approx(.1)
    assert outcome["first_3s_radio_start"] == pytest.approx(3.9)
    assert outcome["final_joint_window_start"] == pytest.approx(5.3)
    assert outcome["first_joint_hold_end"] == pytest.approx(8.3)


def test_component_merge_identifies_the_actual_pivotal_edge():
    a = np.array([[0,1,0], [1,0,0], [0,0,0]], dtype=np.uint8)
    b = a.copy()
    b[1,2] = b[2,1] = 1

    def row(graph):
        return dict(observed=[[0,[0,0]],[1,[1,0]],[2,[2,0]]],
                    comm_hex=np.packbits(graph).tobytes().hex(),
                    relay={"2":[1,[1,0]]}, probe=None)

    result = threshold_merge({10:row(a), 11:row(b)}, 1.1)
    assert result["largest_before"] == 2
    assert result["largest_after"] == 3
    assert result["gained_intercomponent_edges"] == [
        dict(uavs=[1,2], distance=1., largest_without_edge=2)]
    assert result["largest_if_helper_isolated"] == {1:1}


def test_retarget_forecast_exposes_a_split_despite_equal_largest_component():
    positions = {0:[0,0], 1:[1,0], 2:[2,0], 3:[4,0]}
    stay = forecast_retarget(positions, 1, [1,0], 1.1)
    move = forecast_retarget(positions, 1, [3,0], 1.1)
    assert stay["component_sizes"] == move["component_sizes"] == [3, 1]
    assert stay["preserved_nonhelpers"] and not move["preserved_nonhelpers"]
