from types import SimpleNamespace as NS
import json
import numpy as np
from scripts.gppo_motion_capture import capture_motion_frame,json_value,append_motion_frame

class GuardedRuntime:
    current_step=12
    dt=.1
    robots=[NS(robot_id=7,position=np.array([1.,2.]),heading=.5,travel_distance=3.)]
    _motion_feedback={7:{'moved_m':.2}}
    def __getattr__(self,name):
        raise AssertionError('Unexpected runtime access: '+name)

def fixture():
    recovery=NS(frame=NS(describe=lambda:{'cell_size':.4}),cells={(2,3):0},
        prefixes={7:[np.array([2.,2.])]},prefix_headings={7:.2},anchors={7:np.array([3.,3.])},failed={})
    return NS(search_assignments={100:7},relay_assignments={},
        heat_by_id={100:NS(position=np.array([3.,3.]))},_search_recovery=recovery)

def test_capture_serializes_supported_actual_state_without_hidden_runtime_access(tmp_path):
    observations={7:{'position':np.array([1.,2.]),'heading':.5,
        'visible_occupancy':np.array([[2,3,0]]),'hidden_target_truth':'never serialize'}}
    frame=capture_motion_frame(GuardedRuntime(),fixture(),observations,[(np.array([2.,2.]),.2)])
    file=tmp_path/'frames.jsonl'
    append_motion_frame(file,frame)
    restored=json.loads(file.read_text())
    assert restored['actual_commands']['7']['waypoint']==[2.,2.]
    assert restored['observed_recovery']['cells']==[[2,3,0]]
    assert 'hidden_target_truth' not in restored['selected_own_observations']['7']
    assert restored['physical_poses_diagnostic_only']['7']['travel_distance']==3.

def test_capture_includes_only_assigned_own_observations():
    frame=capture_motion_frame(GuardedRuntime(),fixture(),{7:{'heading':.5},8:{'heading':1.}})
    assert set(frame['selected_own_observations'])=={7}
