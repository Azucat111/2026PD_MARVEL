from pathlib import Path
import sys,json,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_travel import extended_observed_travel_time_matrix
from integrations.gppo.observed_graph import observed_next_hop
from integrations.gppo.task_graph import UAVState,Subtask,TaskType
from utils.geometry import GeometryFrame
p=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt4/episode_002/gppo/mission_frames.jsonl'
fs=[json.loads(x) for x in p.read_text().splitlines()];f=next(x for x in fs if x['step']==790);meta=f['observed_recovery']['frame'];r=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));r.observe({int(k):v for k,v in f['selected_own_observations'].items()});r.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
g=f['observed_graphs_diagnostic_only']['21'];coords=np.array(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
pos=np.array(f['selected_own_observations']['21']['position']);goal=np.array(f['public_relay_assignments']['6']['anchor']);d={}
cost=extended_observed_travel_time_matrix([UAVState(21,position=pos,velocity=1.)],[Subtask(200006,TaskType.RELAY,2,active=True,position=goal)],{21:agent},r,d)
report=dict(step=790,position=pos.tolist(),goal=goal.tolist(),old_next_hop=str(observed_next_hop(agent,goal)),extended_cost=cost.tolist(),diagnostics=str(d),graph_nodes=len(coords),captured_cells=len(r.cells),motion_samples=[dict(step=x['step'],position=x['selected_own_observations']['21']['position'],waypoint=x['actual_commands']['21']['waypoint']) for x in fs if 400<=x['step']<=790 and '21' in x['selected_own_observations']],scope='Existing saved sensor labels and own graph only; no true obstacle map consulted.')
print(json.dumps(report))
out=ROOT/'artifacts/extended_relay_cycle_61134_20261010_attempt1';out.mkdir(exist_ok=True)
with (out/'existing_cost_route_diagnosis.json').open('x') as s:json.dump(report,s,indent=2)
