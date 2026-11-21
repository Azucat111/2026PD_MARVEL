"""Offline public radio-intent probe; actual collision map only diagnoses outputs."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.observed_graph import observed_next_hop
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from utils.dynamics_models import KinematicSimple
pair=ROOT/'artifacts/extended_corrected_units_joint_30_dynamic_9162_20261010_attempt2/gppo'
frames=[json.loads(s) for s in (pair/'mission_frames.jsonl').read_text().splitlines() if s.strip().endswith('}')]
f=frames[-1];step=f['step'];cfg=json.loads((pair/'config.json').read_text());meta=f['observed_recovery']['frame']
rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(c,r):v for c,r,v in f['observed_recovery']['cells']}
om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);om.step(step,.1)
runtime=types.SimpleNamespace(current_step=step,obstacles=om,geometry_mode='extended');hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
trace=json.loads((pair/'joint_lifecycle_trace.json').read_text())[-1]
positions={int(uid):np.asarray(p) for uid,p in trace['public_delayed_positions'].items()}
assignments={int(t):(int(a['helper']),np.asarray(a['anchor'])) for t,a in trace['relay_assignments'].items()}
from integrations.gppo.mesh_relay import mesh_components
s=GPPOTaskScheduler.__new__(GPPOTaskScheduler);s.runtime=runtime;s._search_recovery=rec;s.safety=types.SimpleNamespace(hazards=hazards)
b=ObservedMeshRelayDemandBuilder(comm_range=20,max_hops=2,max_demands=2);b.anchor_candidates_provider=s._observed_relay_anchor_candidates
result=b.build(positions,source_task_id='T3_relay')
rows=[]
for slot in result.slots:
 goal=np.asarray(slot.position)
 for helper,p in positions.items():
  if helper in slot.forbidden_uav_ids or helper in [u for u,_ in assignments.values()]:continue
  moved={**positions,helper:goal};lcc=max(map(len,mesh_components(moved,20)))/len(moved)
  if lcc<.9:continue
  public_segment=rec.segment_safe(p,goal) and not hazards.segment_intersects_warning_buffer(p,goal,step=step)
  if not public_segment:continue
  delta=goal-p;length=float(np.linalg.norm(delta));states=[p+delta*t for t in np.linspace(0,1,int(np.ceil(length/.1))+1)]
  collisions=[om.check_collision(x,radius=.2)[0] for x in states]
  peers=[uid for uid,xy in positions.items() if uid!=helper and min(float(np.linalg.norm(x-xy)) for x in states)<.5-1e-9]
  rows.append(dict(helper=helper,goal=goal.tolist(),current_position=p.tolist(),radio_lcc_forecast=lcc,observed_continuous_segment_clear=public_segment,physical_sampled_collision=any(collisions),samples=len(states),stationary_public_peer_conflicts=peers,task_target_uid=slot.source_entity_id))
report=dict(step=step,component_sizes=[len(g) for g in mesh_components(positions,20)],current_helper_count=len(assignments),configured_helper_cap=2,additional_observed_candidates=rows,geometric_test_point_33_26=dict(observed_label=rec.cells.get((33,26)),oracle_collision=om.check_collision(np.array([33.,26.]),radius=.2)),scope='Actual captured delayed public radio poses and observed FREE labels produce added-helper candidates. True map only checks diagnostic sampled trajectories; not actual execution. Stationary peer check is not a future moving-peer certificate.',acceptance_claim=False)
print(json.dumps(report),flush=True)
( ROOT/'artifacts/extended_dynamic_relay_warning_repair_20261010_attempt1/additional_helper_feasibility.json').write_text(json.dumps(report,indent=2))
