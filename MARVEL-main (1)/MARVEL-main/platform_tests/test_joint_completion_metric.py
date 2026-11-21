import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gppo_joint_completion_metric import joint_completion_metric

def samples():
 return [dict(time=i*.1,connectivity=1.,nonrelay_complete=i>=10,safety_active=[],safety_records=1,escapes=1,no_fly_violations=0,collisions=0) for i in range(1,61)]

def test_requires_real_sustained_full_mission_conditions():
 r=joint_completion_metric(samples(),6.)
 assert r['joint_success'] and r['secondary_joint_completion_seconds']==4.

def test_radio_drop_resets_continuous_window():
 rows=samples();rows[20]['connectivity']=.89
 r=joint_completion_metric(rows,6.)
 assert r['joint_success'] and r['secondary_joint_completion_seconds']==5.2

def test_late_collision_invalidates_early_completion():
 rows=samples();rows[-1]['collisions']=1
 r=joint_completion_metric(rows,6.)
 assert not r['joint_success'] and r['secondary_joint_completion_seconds']==6.

def test_unresolved_safety_is_censored():
 rows=samples()
 for r in rows:r['escapes']=0;r['safety_active']=[2]
 assert not joint_completion_metric(rows,6.)['joint_success']

def test_incomplete_physical_trace_is_not_scored():
 with pytest.raises(ValueError,match='ordered'):joint_completion_metric(samples()[::2],6.)
 with pytest.raises(ValueError,match='Full'):joint_completion_metric(samples()[:-1],6.)
