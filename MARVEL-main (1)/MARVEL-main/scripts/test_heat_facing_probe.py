from types import SimpleNamespace as NS
import numpy as np
from utils.geometry import GeometryFrame
from utils.dynamics_models import KinematicSimple
from utils.sensor_models import IdealSensor
from integrations.gppo.search_recovery import ObservedSearchRecovery
from heat_facing_probe import heat_facing_action
def fixture():
 rec=ObservedSearchRecovery(GeometryFrame(0,0,1,30,30))
 rec.cells={(c,r):0 for c in range(30) for r in range(30)}
 robot=NS(robot_id=0,position=np.array([10.5,10.5]),heading=90.,velocity=1.)
 return rec,robot
def test_actual_sensor_gain_without_changing_actual_translation():
 rec,r=fixture();anchor=np.array([16.5,10.5]);command=(np.array([10.5,12.5]),90.)
 altered=heat_facing_action(r,anchor,command,rec,3.5)
 assert altered[1]==0.
 model=KinematicSimple();old,_=model.step(r.position,*command[:1],r.heading,command[1],1.,.1)
 new,_=model.step(r.position,altered[0],r.heading,altered[1],1.,.1)
 np.testing.assert_array_equal(old['position'],new['position'])
 sensor=IdealSensor(dict(fov=120.,range=10.,los_occlusion=True),frame=rec.frame);grid=np.zeros(rec.frame.shape,dtype=np.uint8)
 target=tuple(rec.frame.world_to_cell_floor(anchor))
 assert target not in set(map(tuple,sensor._visible_cells(old['position'],old['heading'],grid)))
 assert target in set(map(tuple,sensor._visible_cells(new['position'],new['heading'],grid)))
def test_frontier_sensing_heading_is_preserved():
 rec,r=fixture();rec.sensing_uids.add(0);a=(np.array([10.5,12.5]),90.)
 assert heat_facing_action(r,[16.5,10.5],a,rec,3.5) is a
def test_unknown_heat_corridor_is_preserved():
 rec,r=fixture();rec.cells={};a=(np.array([10.5,12.5]),90.)
 assert heat_facing_action(r,[16.5,10.5],a,rec,3.5) is a
def test_stationary_command_preserves_no_yaw():
 rec,r=fixture();a=(r.position.copy(),90.)
 assert heat_facing_action(r,[16.5,10.5],a,rec,3.5) is a
def test_service_scan_preserved():
 rec,r=fixture();a=(np.array([10.5,12.5]),90.)
 assert heat_facing_action(r,[11.5,10.5],a,rec,3.5) is a
