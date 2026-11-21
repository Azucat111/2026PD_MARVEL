from pathlib import Path
import json,sys,types,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_travel import extended_observed_next_hop
from utils.geometry import GeometryFrame
old=types.ModuleType('integrations.gppo.goal_region_before');old.__package__='integrations.gppo'
exec(compile((ROOT/'artifacts/extended_search_goal_region_20261010_attempt1/extended_travel.py.before').read_text(),'before','exec'),old.__dict__)
results=[]
for attempt,step,uid,target in [(4,480,21,'6'),(6,560,15,'25')]:
 pair=ROOT/f'artifacts/extended_gppo_joint_batch_20261010_attempt{attempt}/episode_002/gppo'
 f=next(json.loads(x) for x in (pair/'mission_frames.jsonl').read_text().splitlines() if json.loads(x)['step']==step)
 meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']};rec.observe({int(k):v for k,v in f['selected_own_observations'].items()})
 g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.array(g['node_coords']);adj=np.ones((len(coords),len(coords)))
 for a,b in g['zero_adjacency_edges']:adj[a,b]=0
 agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']));pose=rec.observed_positions[uid];goal=np.array(f['public_relay_assignments'][target]['anchor'])
 a=old.extended_observed_next_hop(uid,pose,goal,agent,rec);b=extended_observed_next_hop(uid,pose,goal,agent,rec)
 assert (a is None)==(b is None)
 if a is not None:np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
 results.append(dict(uid=uid,step=step,identical_default_radius_zero=True,next_hop=None if b is None else b[0].tolist()))
frozen=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth');sha=hashlib.sha256(frozen.read_bytes()).hexdigest();assert sha=='57abf685f7a8226dcf71e1441333bf74da6e228e423f1bb0034e110975e3fcd9'
result=dict(cases=results,frozen_sha256=sha,passed=True,scope='Captured Relay boundary route equivalence only; earlier continuous165/136s results retained.',acceptance_claim=False)
with (ROOT/'artifacts/extended_search_goal_region_20261010_attempt1/relay_default_equivalence.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
