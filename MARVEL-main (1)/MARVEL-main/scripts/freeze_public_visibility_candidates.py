"""Public-only visibility ranking using the existing exact IdealSensor and prior geometry."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame,iter_cells_in_radius
from utils.sensor_models import IdealSensor
OUT=ROOT/'artifacts/extended_public_visibility_preflight_20261015_attempt24'
plan=json.loads((OUT/'plan.json').read_text());rankings=[]
for seed in plan['training_seeds']:
 state=json.loads((OUT/'data'/f'seed_{seed}'/'public_state.json').read_text());ctx=state['public_visibility'];g=state['graph'];f=ctx['frame']
 frame=GeometryFrame(f['origin'][0],f['origin'][1],f['cell_size'],f['width_cells'],f['height_cells'],mode=f['mode'])
 known={(c,r):v for c,r,v in ctx['observed_cells']};potential=np.zeros(frame.shape,dtype=np.uint8)
 for (c,r),v in known.items():
  if v==1:potential[r,c]=1
 priors=[]
 for position in ctx['task_positions']:
  if position is None:priors.append(set());continue
  prior={tuple(map(int,cell)) for cell in iter_cells_in_radius(frame,position,ctx['prior_radius']) if known.get(tuple(map(int,cell)))!=1 and np.linalg.norm(frame.cell_to_world(cell)-position)<=ctx['prior_radius']}
  priors.append(prior)
 visible=[]
 for uav in ctx['uavs']:
  if uav['heading'] is None:visible.append(None);continue
  sensor=IdealSensor(dict(fov=uav['fov'],range=uav['sensor_range'],los_occlusion=True),frame=frame)
  visible.append(set(map(tuple,sensor._visible_cells(np.asarray(uav['position']),uav['heading'],potential))))
 options=[]
 for ti,tid in enumerate(g['task_ids']):
  for ui,uid in enumerate(g['uav_ids']):
   if g['mask'][ti][ui] or visible[ui] is None or not priors[ti]:continue
   hits=visible[ui]&priors[ti];fraction=len(hits)/len(priors[ti]);known_fraction=sum(known.get(c)==0 for c in hits)/len(priors[ti])
   options.append(dict(task=tid,uid=uid,action=ti*len(g['uav_ids'])+ui,potential_prior_fraction=fraction,known_free_prior_fraction=known_fraction,prior_cells=len(priors[ti]),observed_cost=g['raw_edge_features'][ti][ui][2]))
 # Freeze candidates solely from current public state, before branch outcomes.
 options.sort(key=lambda o:(-o['potential_prior_fraction'],-o['known_free_prior_fraction'],o['observed_cost'],o['task'],o['uid']))
 baseline=state['top_public_options'][0];baseline=next(o for o in options if (o['task'],o['uid'])==(baseline['task'],baseline['uid']))
 chosen=[baseline]+[o for o in options if o['action']!=baseline['action']][:3]
 old_keys=[(o['task'],o['uid']) for o in state['top_public_options']]
 rankings.append(dict(seed=seed,candidates=chosen,old_top4_actions=old_keys,new_alternates=sum((o['task'],o['uid']) not in old_keys for o in chosen),actor_inputs='public task priors, delayed positions/headings, actual per-type sensor settings, observed occupied cells only',unknown_transparency='Potential information only; never flight certification'))
result=dict(rankings=rankings,training_labels_seen=False,optimizer_updates=0,goal_complete=False,acceptance_claim=False)
(OUT/'public_candidates_frozen.json').write_text(json.dumps(result,indent=2));print(json.dumps([dict(seed=r['seed'],new_alternates=r['new_alternates'],candidates=[dict(task=o['task'],uid=o['uid'],potential=o['potential_prior_fraction'],known=o['known_free_prior_fraction']) for o in r['candidates']]) for r in rankings]),flush=True)
