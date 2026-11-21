from types import SimpleNamespace as NS
import numpy as np
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame

def scheduler():
    sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    sched._search_recovery=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80))
    sched.runtime=NS(obstacles=NS(frame=sched._search_recovery.frame))
    sched._search_recovery.cells={(c,r):0 for c in range(80) for r in range(80)}
    return sched

def test_extended_relay_holds_real_anchor_instead_of_resuming_exploration():
    sched=scheduler();robot=NS(robot_id=0,position=np.array([20.5,20.5]),heading=30.)
    action=sched._relay_action(robot,robot.position.copy(),None,None,(np.array([16.,16.]),120.))
    np.testing.assert_array_equal(action[0],robot.position);assert action[1]==robot.heading

def test_extended_relay_attaches_to_nearby_observed_anchor_from_real_pose():
    sched=scheduler();robot=NS(robot_id=0,position=np.array([19.7,20.5]),heading=0.)
    action=sched._relay_action(robot,np.array([20.5,20.5]),None,None,(np.array([16.,16.]),120.))
    np.testing.assert_array_equal(action[0],[20.5,20.5])
    assert sched._search_recovery.segment_safe(robot.position,action[0])

def test_extended_relay_route_crossing_observed_wall_cannot_execute_unchecked():
    sched=scheduler();sched._search_recovery.cells.update({(22,r):1 for r in range(80)})
    robot=NS(robot_id=0,position=np.array([20.5,20.5]),heading=0.)
    action=sched._relay_action(robot,np.array([25.5,20.5]),None,(np.array([24.,20.]),0),(np.array([20.5,22.]),90.))
    assert sched._search_recovery.segment_safe(robot.position,action[0])
    assert action[0][0]<22.

def test_native_relay_retains_frozen_route_and_base_fallback():
    sched=scheduler();sched._search_recovery=None
    robot=NS(robot_id=0,position=np.array([10.,10.]),heading=0.)
    base=(np.array([20.,20.]),90.)
    assert sched._relay_action(robot,[10.,10.],None,None,base) is base
    action=sched._relay_action(robot,[10.,10.],None,(np.array([14.,10.]),0),base)
    np.testing.assert_array_equal(action[0],[14.,10.]);assert action[1]==0.
