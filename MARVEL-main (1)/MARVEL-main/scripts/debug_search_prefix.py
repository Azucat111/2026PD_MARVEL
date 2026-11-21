"""Two-mission diagnostic capture for the new observed-prefix hold."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import torch
torch.set_num_threads(1)
from benchmark_live_gppo import build
from integrations.gppo.observed_graph import observed_next_hop
runtime,adapter,activation=build(60,0,policy={"device":"cuda","batch_inference":True})
observations=runtime._get_observations()
for mission in (1,2):
    actions=adapter.get_actions(observations)
    if mission==2:break
    for _tick in range(10): observations,_=runtime.step(actions)
controller=adapter.scheduler._search_recovery
result={"activation":activation,"cases":[]}
for uid in (0,43):
    robot=runtime.robots[uid];agent=adapter.agents[uid]
    hid=next(h for h,u in adapter.scheduler.search_assignments.items() if u==uid)
    anchor=adapter.scheduler.heat_by_id[hid].position
    route=observed_next_hop(agent,anchor)
    candidates=[]
    for idx in np.unique(np.asarray(agent.neighbor_indices).astype(int)):
        target=np.asarray(agent.node_coords[idx])
        length=float(np.linalg.norm(target-robot.position))
        if length>.1 and length<=6:
            bad={}
            for offset in ((0,0),(.2,0),(-.2,0),(0,.2),(0,-.2)):
                for cell in controller._ray_cells(robot.position+offset,target+offset):
                    if controller.cells.get(cell)!=0:bad[cell]=controller.cells.get(cell)
            candidates.append({"index":int(idx),"target":target.tolist(),"length":length,
                "closing":float(np.linalg.norm(robot.position-anchor)-np.linalg.norm(target-anchor)),
                "unsafe_cells":[[int(c),int(r),v] for (c,r),v in bad.items()]})
    result["cases"].append({"uid":uid,"position":robot.position.tolist(),"heading":robot.heading,
        "anchor":np.asarray(anchor).tolist(),"selected":np.asarray(actions[uid][0]).tolist(),
        "preferred":None if route is None else np.asarray(route[0]).tolist(),
        "neighbors":candidates,"feedback":observations[uid]["motion_feedback"],
        "observed_cells":[[int(c),int(r),v] for (c,r),v in controller.cells.items()
            if np.linalg.norm(np.array([c,r])-robot.position)<12]})
out=ROOT/"artifacts"/"implemented_search_recovery_20261010_attempt3"/"initial_prefix_debug.json"
with out.open("x") as f:json.dump(result,f,indent=2)
for case in result["cases"]:
    print(json.dumps({k:v for k,v in case.items() if k!="observed_cells"},indent=2))
