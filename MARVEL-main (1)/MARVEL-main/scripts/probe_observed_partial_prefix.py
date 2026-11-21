from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame
data=json.loads((ROOT/"artifacts/implemented_search_recovery_20261010_attempt3/initial_prefix_debug.json").read_text())
for case in data["cases"]:
    if case["uid"] != 0:continue
    controller=ObservedSearchRecovery(GeometryFrame.extended(width_cells=500,height_cells=500))
    controller.cells={(c,r):v for c,r,v in case["observed_cells"]}
    pos=np.array(case["position"]);anchor=np.array(case["anchor"])
    print("case",case["uid"])
    for node in case["neighbors"]:
        target=np.array(node["target"]);vector=(target-pos)/np.linalg.norm(target-pos)
        for length in (1.,.5,.25,.125):
            prefix=pos+vector*length
            if controller.segment_safe(pos,prefix):
                print(json.dumps({"node":node["target"],"prefix":prefix.tolist(),"length":length,
                    "closing":float(np.linalg.norm(pos-anchor)-np.linalg.norm(prefix-anchor))}))
