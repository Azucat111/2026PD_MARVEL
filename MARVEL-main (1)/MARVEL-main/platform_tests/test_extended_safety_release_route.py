from types import SimpleNamespace as NS
import numpy as np
from integrations.gppo.extended_safety_motion import ObservedMotionSafetyLayer
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame
from utils.dynamics_models import KinematicSimple

def fixture():
 start=np.array([16.767605602289617,31.54009878744565]);robot=NS(robot_id=0,position=start.copy(),heading=135.0584217381396,velocity=1.,robot_type='explorer')
 obstacle=NS(obstacle_id='warning',position=[20.,20.],spawn_step=0,initial_radius=4.,expansion_rate=0.,max_radius=4.)
 runtime=NS(current_step=400,dt=.1,robots=[robot],dynamics=KinematicSimple(),dynamics_by_type={},obstacles=NS(dynamic_obstacles=[obstacle]))
 rec=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80));rec.cells={(c,r):0 for c in range(80) for r in range(80)};rec.nearby_positions[0]=[np.array([17.,32.]),np.array([16.266044000756263,31.363153651866778])]
 layer=ObservedMotionSafetyLayer(runtime,recovery=rec)
 return layer,robot,rec

def test_peer_choke_can_take_temporary_clearance_dip_and_reach_original_safe_margin():
 layer,robot,rec=fixture();start=robot.position.copy();minimum=1e9
 for tick in range(40):
  point,heading=layer.select_action(robot,step=400+tick);assert rec.segment_safe(robot.position,point)
  assert not layer.hazards.segment_intersects_warning_buffer(robot.position,point,step=400+tick)
  delta=point-robot.position;d2=np.dot(delta,delta)
  for peer in rec.nearby_positions[0]:
   closest=robot.position+(np.clip(np.dot(peer-robot.position,delta)/d2,0,1) if d2 else 0)*delta
   assert np.linalg.norm(closest-peer)>=.5-1e-9
  update,_=layer.runtime.dynamics.step(robot.position,point,robot.heading,heading,dt=.1);robot.position=update['position'];robot.heading=update['heading'];minimum=min(minimum,layer.hazards.risk_at(robot.position,step=400+tick).clearance)
  if layer.hazards.is_safe(robot.position,step=400+tick):break
 assert layer.hazards.is_safe(robot.position,step=440)
 assert minimum<layer.hazards.risk_at(start,step=400).clearance

def test_already_safe_active_uav_holds_for_unchanged_stable_release():
 layer,robot,rec=fixture();robot.position=np.array([18.,33.]);robot.heading=123.
 point,heading=layer.select_action(robot,step=400)
 np.testing.assert_array_equal(point,robot.position);assert heading==123.

def test_unknown_release_route_is_never_admitted():
 layer,robot,rec=fixture();rec.cells.clear()
 point,heading=layer.select_action(robot,step=400)
 np.testing.assert_array_equal(point,robot.position);assert heading==robot.heading

def test_cached_release_leg_rechecks_new_peer():
 layer,robot,rec=fixture();point,_=layer.select_action(robot,step=400)
 assert np.linalg.norm(point-robot.position)>.1
 rec.nearby_positions[0].append((point+robot.position)/2)
 next_point,_=layer.select_action(robot,step=401)
 assert not np.array_equal(point,next_point)
