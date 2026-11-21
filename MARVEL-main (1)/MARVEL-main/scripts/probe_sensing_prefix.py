"""Diagnostic hypothesis: legal graph-direction sensing prefixes with independent
desired heading, a control pair already supported by KinematicSimple.
No production monkeypatch or file edit. Known cells are sensor/accepted-motion only.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from types import SimpleNamespace
from utils.geometry import GeometryFrame
from utils.sensor_models import IdealSensor
from utils.obstacle_manager import ObstacleManager
from utils.scenario_config import load_and_validate_scenario
from utils.dynamics_models import create_dynamics_model
from integrations.gppo.search_recovery import ObservedSearchRecovery
data=json.loads((ROOT/"artifacts/implemented_search_recovery_20261010_attempt3/initial_prefix_debug.json").read_text())
case=next(c for c in data["cases"] if c["uid"]==0)
config=load_and_validate_scenario(ROOT/"configs/scenarios/extended_live_gppo_full.yaml")
ob=ObstacleManager(config["environment"]);ob.load_from_file(config["environment"]["map_file"])
controller=ObservedSearchRecovery(ob.frame)
controller.cells={(c,r):v for c,r,v in case["observed_cells"]}
coords=np.array([n["target"] for n in case["neighbors"]])
agent=SimpleNamespace(node_coords=coords,neighbor_indices=np.arange(len(coords)))
robot=SimpleNamespace(robot_id=0,position=np.array(case["position"]),heading=case["heading"],velocity=1.)
anchor=np.array(case["anchor"])
sensor=IdealSensor({"fov":120,"range":10,"los_occlusion":True},ob.frame)
dyn=create_dynamics_model(config["dynamics"]["params"])
rows=[];grid=ob.get_occupancy_grid()
for mission in range(1,31):
    before=robot.position.copy()
    target=controller.select(0,robot.position,anchor,agent,case["preferred"],[96.,112.])
    kind="normal"
    if np.linalg.norm(target-robot.position)<.1:
        candidates=[]
        for order,node in enumerate(coords):
            delta=node-robot.position;distance=np.linalg.norm(delta)
            if distance<.1:continue
            for length in (1.,.5,.25,.125):
                if length>=distance:continue
                prefix=robot.position+length*delta/distance
                if controller.segment_safe(robot.position,prefix):
                    closing=np.linalg.norm(robot.position-anchor)-np.linalg.norm(prefix-anchor)
                    candidates.append((-closing,-length,order,prefix));break
        if candidates:
            target=min(candidates,key=lambda x:x[:3])[3];kind="sensing_retreat"
        else:kind="hold"
    # Independent heading is a supported action; no stationary rotation.
    desired=float(np.degrees(np.arctan2(*(anchor-robot.position)[::-1]))%360)
    for tick in range(10):
        state,_=dyn.step(robot.position,target,robot.heading,desired,1.,.1)
        assert not ob.check_collision(state["position"],radius=.2)[0]
        robot.position=state["position"];robot.heading=state["heading"]
        obs=sensor.sense(robot,grid,[(0,robot.position)])
        obs["visible_occupancy"]=[[int(c),int(r),int(grid[r,c])] for c,r in obs["visible_cells"]]
        obs["occupancy_frame"]=ob.frame.describe()
        obs["motion_feedback"]={"step":mission*10+tick,"accepted":True,
            "grid_probes_clear":True,"proposal":robot.position.tolist(),"target":target.tolist()}
        controller.observe({0:obs})
    rows.append({"mission":mission,"kind":kind,"target":target.tolist(),
        "position":robot.position.tolist(),"heading":robot.heading,
        "moved":float(np.linalg.norm(robot.position-before)),
        "distance":float(np.linalg.norm(robot.position-anchor))})
print(json.dumps(rows,indent=2))
out=ROOT/"artifacts/implemented_search_recovery_20261010_attempt3/sensing_prefix_hypothesis.json"
with out.open("x") as f:json.dump({"supported_motion":"KinematicSimple waypoint + independent desired_heading",
    "all_300_ticks_collision_free":True,"rows":rows},f,indent=2)
