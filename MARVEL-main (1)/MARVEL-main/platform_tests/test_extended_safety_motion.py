from types import SimpleNamespace as NS
import numpy as np
from integrations.gppo.extended_safety_motion import ObservedMotionSafetyLayer
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame
from utils.dynamics_models import KinematicSimple

def layer(known=True,neighbors=None):
    robot=NS(robot_id=54,position=np.array([21.559838546,20.412908]),heading=348.320828,velocity=1.,robot_type='explorer')
    obstacle=NS(obstacle_id='warning',position=[20.,20.],spawn_step=1,initial_radius=0.,expansion_rate=.15,max_radius=4.)
    runtime=NS(current_step=1,dt=.1,robots=[robot],dynamics=KinematicSimple(),dynamics_by_type={},obstacles=NS(dynamic_obstacles=[obstacle]))
    recovery=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80))
    if known:recovery.cells={(c,r):0 for c in range(80) for r in range(80)}
    result=ObservedMotionSafetyLayer(runtime,recovery=recovery)
    if neighbors is not None:result.bind_marvel_agents([NS(id=54,node_coords=np.array(neighbors),neighbor_indices=np.arange(len(neighbors)),current_index=0)])
    return result,robot,recovery

def test_escape_does_not_choose_far_endpoint_through_hazard_center():
    result,robot,recovery=layer(neighbors=[[12.,12.],[24.,20.]])
    target,heading=result.select_action(robot,step=1)
    before=np.linalg.norm(robot.position-[20.,20.])
    update,_=result.runtime.dynamics.step(robot.position,target,robot.heading,heading,dt=.1)
    assert np.linalg.norm(update['position']-[20.,20.])>before
    assert recovery.segment_safe(robot.position,target)

def test_public_radial_prefix_is_supported_without_observed_graph_neighbor():
    result,robot,recovery=layer(neighbors=[[12.,12.]])
    target,_=result.select_action(robot,step=1)
    assert np.linalg.norm(target-robot.position)>.1
    assert np.dot(target-robot.position,robot.position-[20.,20.])>0.

def test_unknown_static_prefix_fails_closed_and_hold_cannot_yaw():
    result,robot,_=layer(known=False,neighbors=[[24.,20.]])
    target,heading=result.select_action(robot,step=1)
    np.testing.assert_array_equal(target,robot.position)
    assert heading==robot.heading

def test_detected_peer_blocks_outward_prefix():
    result,robot,recovery=layer(neighbors=[[12.,12.]])
    direction=(robot.position-[20.,20.])/np.linalg.norm(robot.position-[20.,20.])
    recovery.nearby_positions[54]=[robot.position+.5*direction]
    target,_=result.select_action(robot,step=1)
    delta=target-robot.position;peer=recovery.nearby_positions[54][0]
    fraction=np.clip(np.dot(peer-robot.position,delta)/np.dot(delta,delta),0.,1.) if np.dot(delta,delta)>0. else 0.
    assert np.linalg.norm(robot.position+fraction*delta-peer)>=.5-1e-9

def test_crossing_a_second_public_hazard_is_rejected():
    result,robot,recovery=layer(neighbors=[[24.,20.]])
    result.runtime.obstacles.dynamic_obstacles.append(NS(obstacle_id='second',position=[23.,20.],spawn_step=1,initial_radius=0.,expansion_rate=0.,max_radius=4.))
    target,_=result.select_action(robot,step=1)
    for center in (np.array([20.,20.]),np.array([23.,20.])):
        assert np.dot(target-robot.position,robot.position-center)>=-1e-9
    assert recovery.segment_safe(robot.position,target)

def test_physics_runtime_calls_extended_safety_hook_before_motion():
    from utils.simulation_runtime import SimulationRuntime
    from utils.scenario_config import load_and_validate_scenario
    from pathlib import Path
    config=load_and_validate_scenario(Path(__file__).resolve().parents[1]/'configs/scenarios/baseline_exploration.yaml')
    config['environment']['circular_obstacles']=[]
    config['robots']=[{'id_range':[0,0],'type':'explorer','config':{'initial_positions':[[21.6,20.4]],'velocity':1.}}]
    runtime=SimulationRuntime(config);runtime.reset();seen=[]
    def before(actions):
        seen.append(runtime.current_step)
        return [(np.array([24.,20.4]),0.)]
    runtime.high_level_scheduler=NS(before_physics_tick=before)
    runtime.step([(np.array([16.,20.4]),180.)])
    assert seen==[0] and runtime.robots[0].position[0]>21.6


def test_fast_expansion_escape_accounts_for_observed_wall_and_future_front():
    result,robot,recovery=layer(neighbors=[[12.,12.],[12.,28.]])
    recovery.cells={(c,r):int(not(4<=c<24 and 4<=r<34)) for c in range(80) for r in range(80)}
    for step in range(1,40):
        result.runtime.current_step=step
        target,heading=result.select_action(robot,step=step)
        update,_=result.runtime.dynamics.step(robot.position,target,robot.heading,heading,dt=.1)
        assert recovery.segment_safe(robot.position,update['position'])
        risk=result.hazards.risk_at(update['position'],step=step)
        assert risk.clearance>.2
        robot.position=update['position'];robot.heading=update['heading']
