"""Capture one real empty-lattice graph boundary; one mission, no production patch."""
from pathlib import Path
import json,sys
import benchmark_fair_search as bench
import utils.node_manager as node_module
out=bench.ROOT/'artifacts/empty_observed_graph_20261010_attempt2'
out.mkdir(exist_ok=False)
original=node_module.get_updating_node_coords
failures=[]
class CapturedBoundary(BaseException):pass
def capture(location,map_info,*args,**kwargs):
    try:return original(location,map_info,*args,**kwargs)
    except IndexError as error:
        entry={'position':location.tolist(),'origin':[map_info.map_origin_x,map_info.map_origin_y],
            'cell_size':map_info.cell_size,'map':map_info.map.tolist(),'error':str(error)}
        failures.append(entry)
        (out/'failures.json').write_text(json.dumps(failures))
        raise CapturedBoundary() from error
node_module.get_updating_node_coords=capture
try:result=bench.run(out,'gppo',30,78901,20,0,True,False)
except CapturedBoundary:pass
finally:node_module.get_updating_node_coords=original
print(json.dumps({'captured_failures':len(failures),'positions':[f['position'] for f in failures]}),flush=True)
