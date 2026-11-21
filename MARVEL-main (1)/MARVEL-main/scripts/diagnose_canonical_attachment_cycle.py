"""Captured observed-only motion decision; true map is validation oracle only."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.extended_travel import extended_observed_next_hop,extended_observed_travel_time_matrix
from integrations.gppo.observed_graph import observed_next_hop
from integrations.gppo.safety_layer import SharedHazardAdapter
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from utils.dynamics_models import KinematicSimple
PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt6/episode_002/gppo'
OUT=ROOT/'artifacts/extended_relay_attachment_cycle_61166_20261010_attempt1/diagnosis.json'
frames=[json.loads(s) for s in (PAIR/'mission_frames.jsonl').read_text().splitlines()]
f=next(x for x in frames if x['step']==560);prior=next(x for x in frames if x['step']==550);uid=15
meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
rec.observe({int(k):o for k,o in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
for field in ('prefixes','prefix_headings','anchors'):
    setattr(rec,field,{int(k):float(v) if field=='prefix_headings' else np.array(v) for k,v in prior['observed_recovery'][field].items()})
rec.failed={int(k):[(np.array(t),np.array(p)) for t,p in values] for k,values in prior['observed_recovery']['failed'].items()}
g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
own=f['selected_own_observations'][str(uid)];robot=types.SimpleNamespace(robot_id=uid,position=np.array(own['position']),heading=own['heading'],current_task='relay')
anchor=np.array(next(a['anchor'] for a in f['public_relay_assignments'].values() if a['helper']==uid))
cfg=json.loads((PAIR/'config.json').read_text());om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);om.step(560,.1)
runtime=types.SimpleNamespace(current_step=560,obstacles=om,geometry_mode='extended',robots=[robot]);hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
filtered=lambda a,b:not hazards.segment_intersects_warning_buffer(a,b,step=560)
details={};hop=extended_observed_next_hop(uid,robot.position,anchor,agent,rec,segment_filter=filtered,route_details=details)
sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler);sched.runtime=runtime;sched._search_recovery=rec;sched.search_reach_radius=3.5;sched.safety=types.SimpleNamespace(hazards=hazards)
rough=observed_next_hop(agent,anchor);action=sched._relay_action(robot,anchor,agent,rough,(robot.position.copy(),robot.heading))
np.testing.assert_allclose(action[0],f['actual_commands'][str(uid)]['waypoint'])
state,_=KinematicSimple().step(robot.position,action[0],robot.heading,action[1],dt=.1);collision,reason=om.check_collision(state['position'],radius=.2)
result=dict(uid=uid,physical_step=560,position=robot.position.tolist(),goal=anchor.tolist(),heading=robot.heading,
    observed_dijkstra_hop=None if hop is None else hop[0].tolist(),details=details,
    upstream_observed_graph_hop=None if rough is None else rough[0].tolist(),
    reproduced_command=action[0].tolist(),reproduced_heading=float(action[1]),original_command_exactly_reproduced=True,
    first_actual_tick=state['position'].tolist(),first_tick_oracle_collision=collision,collision_reason=reason,
    known_segment_safe=rec.segment_safe(robot.position,action[0]),public_warning_clear=filtered(robot.position,action[0]),
    current_goal_distance=float(np.linalg.norm(robot.position-anchor)),command_goal_distance=float(np.linalg.norm(action[0]-anchor)),
    goal_footprint_observed_FREE=rec.segment_safe(anchor,anchor),
    route_nodes=len(coords),scope='Current own sensor/graph plus prior550 recovery cache. Upstream MARVEL base action not captured; old authoritative command reproduces with hold placeholder before fallback. Truth obstacle map only validates first physical tick.',acceptance_claim=False)
with OUT.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
