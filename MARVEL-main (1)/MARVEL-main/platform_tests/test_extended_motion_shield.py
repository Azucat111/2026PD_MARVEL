import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.safety_shield import SafetyShield
from utils.simulation_runtime import SimulationRuntime
from utils.scenario_config import load_and_validate_scenario

def test_guard_checks_real_tick_crossing_not_remote_destinations():
    shield=SafetyShield(None)
    robots=[SimpleNamespace(robot_id=0,position=np.array([2.,2.])),SimpleNamespace(robot_id=1,position=np.array([2.6,2.]))]
    proposed=[np.array([2.3,2.]),np.array([2.3,2.])]
    allowed=shield.filter_motion(robots,proposed,0)
    accepted=[p if ok else r.position for r,p,ok in zip(robots,proposed,allowed)]
    assert np.linalg.norm(accepted[0]-accepted[1])>=.5
    assert shield.pop_events()

def test_guard_allows_existing_close_pair_to_separate():
    shield=SafetyShield(None)
    robots=[SimpleNamespace(robot_id=0,position=np.array([2.,2.])),SimpleNamespace(robot_id=1,position=np.array([2.3,2.]))]
    assert shield.filter_motion(robots,[np.array([1.9,2.]),np.array([2.4,2.])],0)==[True,True]

def test_runtime_cannot_overlap_when_distinct_far_waypoints_cross():
    config=load_and_validate_scenario(ROOT/'configs/scenarios/baseline_exploration.yaml')
    config['environment']['circular_obstacles']=[]
    config['robots']=[{'id_range':[0,1],'type':'explorer','config':{'initial_positions':[[2.,2.],[2.5,2.]],'velocity':1.}}]
    runtime=SimulationRuntime(config);runtime.reset()
    obs,info=runtime.step([(np.array([5.,2.]),0.),(np.array([1.,2.]),180.)])
    assert not any(e['type']=='uav_uav' for e in info['collisions'])
    assert np.linalg.norm(runtime.robots[0].position-runtime.robots[1].position)>=.5-1e-9
    assert any(o['motion_feedback']['shield_rejected'] for o in obs.values())

def test_shared_remote_waypoint_does_not_block_safe_current_ticks():
    config=load_and_validate_scenario(ROOT/'configs/scenarios/baseline_exploration.yaml')
    config['environment']['circular_obstacles']=[]
    config['robots']=[{'id_range':[0,1],'type':'explorer','config':{'initial_positions':[[2.,2.],[12.,2.]],'velocity':1.}}]
    runtime=SimulationRuntime(config);runtime.reset()
    runtime.step([(np.array([7.,2.]),0.),(np.array([7.,2.]),180.)])
    assert runtime.robots[0].position[0]>2.
    assert runtime.robots[1].position[0]<12.
    assert np.linalg.norm(runtime.robots[0].position-runtime.robots[1].position)>=.5

def test_stopped_uav_is_rechecked_against_other_motion_paths():
    shield=SafetyShield(None)
    robots=[SimpleNamespace(robot_id=i,position=np.array(p,dtype=float)) for i,p in enumerate([[0.,0.],[1.,0.],[2.,0.]])]
    proposals=[np.array([1.,0.]),np.array([2.,0.]),np.array([0.,0.])]
    allowed=shield.filter_motion(robots,proposals,0)
    endpoints=[p if ok else r.position for r,p,ok in zip(robots,proposals,allowed)]
    for i in range(3):
        for j in range(i+1,3):assert np.linalg.norm(endpoints[i]-endpoints[j])>=.5
