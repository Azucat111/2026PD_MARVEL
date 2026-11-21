"""Real-model service scan; an invisible target is not a completed Search."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame
from utils.dynamics_models import KinematicSimple
from utils.sensor_models import IdealSensor

def controller(known=True):
    result=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80))
    if known:result.cells={(c,r):0 for c in range(80) for r in range(80)}
    return result

def scheduler(recovery):
    result=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    result._search_recovery=recovery;result.search_reach_radius=3.5
    return result

def test_service_prefix_clips_existing_waypoint_inside_unchanged_reach_radius():
    recovery=controller()
    result=recovery.select(0,[10.,10.],[10.,10.],None,None,[16.,10.],max_anchor_distance=3.5)
    assert .1<np.linalg.norm(result-[10.,10.])<=3.5
    assert recovery.segment_safe([10.,10.],result)

def test_literal_blind_service_pose_turns_only_through_real_legal_motion():
    recovery=controller();sched=scheduler(recovery)
    robot=SimpleNamespace(robot_id=15,position=np.array([58.65705451630842,17.908144089517894]),heading=10.)
    heat=np.array([58.,20.]);grid=np.zeros((80,80),dtype=np.uint8)
    sensor=IdealSensor({'fov':120,'range':10,'los_occlusion':True},recovery.frame)
    assert sensor._has_line_of_sight(robot.position,(55,21),grid)
    assert (55,21) not in set(map(tuple,sensor._visible_cells(robot.position,robot.heading,grid)))
    action=sched._search_action(robot,heat,None,None,(np.array([60.,20.]),10.),service_scan_heading=107.43756800667182)
    assert recovery.segment_safe(robot.position,action[0])
    assert np.linalg.norm(action[0]-heat)<=3.5
    dynamics=KinematicSimple({'velocity':1.,'yaw_rate':35.})
    seen=False
    for _ in range(10):
        update,_=dynamics.step(robot.position,action[0],robot.heading,action[1],dt=.1)
        robot.position=update['position'];robot.heading=update['heading']
        assert np.linalg.norm(robot.position-heat)<=3.5+1e-9
        seen|=(55,21) in set(map(tuple,sensor._visible_cells(robot.position,robot.heading,grid)))
    assert seen

def test_unknown_service_prefix_never_assumes_stationary_yaw():
    recovery=controller(False);sched=scheduler(recovery)
    robot=SimpleNamespace(robot_id=0,position=np.array([10.,10.]),heading=0.)
    action=sched._search_action(robot,robot.position.copy(),None,None,(np.array([16.,10.]),0.),service_scan_heading=180.)
    update,_=KinematicSimple().step(robot.position,action[0],robot.heading,action[1])
    np.testing.assert_array_equal(update['position'],robot.position)
    assert update['heading']==0.

class PrivateTruth:
    def __getattribute__(self,name):raise AssertionError('Private truth inspected: '+name)

def service_scheduler(native=False,confirmed=False):
    result=scheduler(None if native else controller())
    result.runtime=SimpleNamespace(robots=[SimpleNamespace(robot_id=0,position=np.array([10.,10.]))],
        current_step=69,search_scenario=PrivateTruth(),completed_search_heat_ids=lambda:{0} if confirmed else set())
    result.search_assignments={0:0};result.serviced_heat_ids=set()
    result.heat_by_id={0:SimpleNamespace(position=(10.,10.))}
    result.search_dwell_by_heat={0:5};result.search_service_steps=6
    result.mark_heat_serviced=lambda heat:result.serviced_heat_ids.add(heat)
    return result

def test_extended_unconfirmed_dwell_keeps_search_active_and_truth_private():
    result=service_scheduler()
    events=result._update_search_service()
    assert result.serviced_heat_ids==set()
    assert result.search_assignments=={0:0}
    assert events[0]['type']=='gppo_search_service_unconfirmed'

def test_native_dwell_still_completes_without_sensor_confirmation():
    result=service_scheduler(native=True)
    assert result._update_search_service()[0]['reason']=='dwell'
    assert result.serviced_heat_ids=={0}

def test_confirmed_extended_feedback_can_complete():
    result=service_scheduler(confirmed=True)
    result._update_search_service()
    assert result.serviced_heat_ids=={0}

def test_scan_reference_does_not_rotate_with_the_current_anchor_bearing():
    result=scheduler(controller());result.search_dwell_by_heat={0:0}
    result.runtime=SimpleNamespace(sensor=SimpleNamespace(fov=120.),sensor_by_type={})
    robot=SimpleNamespace(robot_id=0,robot_type='explorer',position=np.array([10.,10.]),heading=0.)
    heat=np.array([10.,10.])
    assert result._service_scan_heading(0,robot,heat)==0.
    result.search_dwell_by_heat[0]=1;robot.position=np.array([11.,10.])
    assert result._service_scan_heading(0,robot,heat)==120.
    result.search_dwell_by_heat[0]=2;robot.position=np.array([9.,10.])
    assert result._service_scan_heading(0,robot,heat)==120.
    robot.heading=120.
    assert result._service_scan_heading(0,robot,heat)==240.


def test_short_real_motion_finishes_a_scan_sector_before_switching():
    result=scheduler(controller());result.search_dwell_by_heat={0:0}
    result.runtime=SimpleNamespace(sensor=SimpleNamespace(fov=120.),sensor_by_type={})
    robot=SimpleNamespace(robot_id=0,robot_type="explorer",position=np.array([10.,10.]),heading=0.)
    heat=robot.position.copy();grid=np.zeros((80,80),dtype=np.uint8)
    sensor=IdealSensor({"fov":120,"range":10,"los_occlusion":True},result._search_recovery.frame)
    dynamics=KinematicSimple({'velocity':1.,'yaw_rate':35.})
    seen=False;headings=[]
    for mission in range(60):
        desired=result._service_scan_heading(0,robot,heat)
        goal=heat+np.array([.125 if mission%2==0 else 0.,0.])
        assert result._search_recovery.segment_safe(robot.position,goal)
        for _ in range(10):
            update,_=dynamics.step(robot.position,goal,robot.heading,desired,dt=.1)
            robot.position=update['position'];robot.heading=update['heading']
            assert np.linalg.norm(robot.position-heat)<=3.5
            seen|=(6,10) in set(map(tuple,sensor._visible_cells(robot.position,robot.heading,grid)))
        headings.append(robot.heading);result.search_dwell_by_heat[0]+=1
    assert seen
    assert headings[2]>headings[1]

def test_extended_missing_confirmation_fails_closed():
    for missing in ("scenario","resolver"):
        result=service_scheduler()
        if missing=="scenario":result.runtime.search_scenario=None
        else:del result.runtime.completed_search_heat_ids
        assert result._update_search_service()[0]["type"]=="gppo_search_service_unconfirmed"
        assert result.serviced_heat_ids==set()


def test_recorded_service_boundary_can_take_observed_prefix_toward_public_heat():
    # Literal pose from new seed61085 failure; observed border y<4/x>=76 is OCC.
    recovery=controller(False)
    recovery.cells={(c,r):int(r<4 or c>=76) for c in range(70,78) for r in range(1,10)}
    sched=scheduler(recovery)
    robot=SimpleNamespace(robot_id=17,position=np.array([72.65982647293558,6.580417355205188]),heading=58.78990397696403)
    heat=np.array([75.,4.])
    # Controlled outward-only graph: the failed trace does not retain its graph neighbors.
    agent=SimpleNamespace(node_coords=np.array([[72.,8.]]),neighbor_indices=np.array([0]))
    action=sched._search_action(robot,heat,agent,None,(np.array([72.,8.]),robot.heading),service_scan_heading=314.26587673526546)
    assert np.linalg.norm(action[0]-robot.position)>.1
    assert recovery.segment_safe(robot.position,action[0])
    assert np.linalg.norm(action[0]-heat)<=3.5
    update,_=KinematicSimple({'velocity':1.,'yaw_rate':35.}).step(robot.position,action[0],robot.heading,action[1],dt=.1)
    assert np.linalg.norm(update['position']-robot.position)>0.
    assert abs((update['heading']-robot.heading+180.)%360.-180.)>0.


def test_service_anchor_fallback_preserves_failed_prefix_exclusion():
    recovery=controller(False)
    recovery.cells={(c,r):int(r<4 or c>=76) for c in range(70,78) for r in range(1,10)}
    sched=scheduler(recovery)
    robot=SimpleNamespace(robot_id=17,position=np.array([72.65982647293558,6.580417355205188]),heading=58.78990397696403)
    heat=np.array([75.,4.]);direction=(heat-robot.position)/np.linalg.norm(heat-robot.position)
    failed=robot.position+direction
    recovery.anchors[17]=heat.copy();recovery.failed[17]=[(failed.copy(),robot.position.copy())]
    action=sched._search_action(robot,heat,None,None,(np.array([72.,8.]),robot.heading),service_scan_heading=314.26587673526546)
    assert np.linalg.norm(action[0]-robot.position)>.1
    assert not np.allclose(action[0],failed)
    assert len(recovery.failed[17])==1
    np.testing.assert_array_equal(recovery.failed[17][0][0],failed)

def test_legal_service_cached_prefix_does_not_invoke_fallback():
    recovery=controller();sched=scheduler(recovery)
    heat=np.array([10.,10.]);prefix=np.array([11.,10.])
    robot=SimpleNamespace(robot_id=0,position=heat.copy(),heading=0.)
    recovery.anchors[0]=heat.copy();recovery.prefixes[0]=prefix.copy()
    original=recovery.select;calls=[]
    def tracked(*args,**kwargs):
        calls.append((args,kwargs));return original(*args,**kwargs)
    recovery.select=tracked
    action=sched._search_action(robot,heat,None,None,(np.array([16.,10.]),0.),service_scan_heading=120.)
    np.testing.assert_array_equal(action[0],prefix)
    assert len(calls)==1 and action[1]==120.
