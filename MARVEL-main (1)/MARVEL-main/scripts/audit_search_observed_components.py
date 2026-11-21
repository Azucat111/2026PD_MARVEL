"""Offline physical oracle vs optimistic observed connectivity, no runtime planner."""
from pathlib import Path
import sys,json,argparse,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.ndimage import label
from utils.simulation_runtime import SimulationRuntime
p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();folder=Path(args.folder)
config=json.loads((folder/'gppo/config.json').read_text());runtime=SimulationRuntime(config)
truth=runtime.obstacles.get_occupancy_grid();shape=truth.shape
snap=json.loads((folder/'snapshot_400.json').read_text());observed=np.full(shape,-1,dtype=int)
for col,row,value in snap['cells']:observed[row,col]=value
oracle,_=label(truth==0,np.ones((3,3)))
optimistic,_=label(observed!=1,np.ones((3,3)))
result=[]
for entry in snap['entries']:
    x,y=entry['position'];col,row=int(np.floor(x)),int(np.floor(y));goal=np.array(entry['goal'])
    item={'uid':entry['uid'],'goal':entry['goal'],'position':entry['position']}
    for name,labels in [('physical_oracle',oracle),('observed_unknown_passable',optimistic)]:
        component=int(labels[row,col]);rr,cc=np.where(labels==component)
        if component==0: item[name]={'invalid_center_cell':True};continue
        centers=np.column_stack([cc+.5,rr+.5]);nearest=float(np.linalg.norm(centers-goal,axis=1).min())
        # Generous center-to-any-point cell allowance + all public target support + sensor range.
        allowance=10+5.5+np.sqrt(2)
        item[name]={'component_cells':len(rr),'minimum_heat_distance_cell_centers_m':nearest,
            'optimistic_detection_allowance_m':allowance,
            'public_heat_support_outside_detection_reach':bool(nearest>allowance)}
    result.append(item)
(folder/'component_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
