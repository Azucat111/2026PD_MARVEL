"""Captured cycle boundary replay: before and certified-route controller, no hidden decision inputs."""
from pathlib import Path
import sys,json,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.observed_graph import observed_next_hop
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from utils.dynamics_models import KinematicSimple
DIR=ROOT/'artifacts/extended_relay_cycle_61134_20261010_attempt1';PAIR=ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt4/episode_002/gppo'
frames=[json.loads(s) for s in (PAIR/'mission_frames.jsonl').read_text().splitlines()];f=next(x for x in frames if x['step']==480);prior=next(x for x in frames if x['step']==470)
cfg=json.loads((PAIR/'config.json').read_text());om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);om.step(480,.1)
runtime=types.SimpleNamespace(current_step=480,obstacles=om,geometry_mode='extended');hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
module=types.ModuleType('integrations.gppo.cycle_before');module.__package__='integrations.gppo';exec(compile((DIR/'scheduler.py.before').read_text(),'before','exec'),module.__dict__);Old=module.GPPOTaskScheduler
results=[]
for kind,cls in [('old',Old),('new',GPPOTaskScheduler)]:
 meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.observe({int(k):o for k,o in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
 for field in ('prefixes','prefix_headings','anchors'):
  setattr(rec,field,{int(uid):float(v) if field=='prefix_headings' else np.array(v) for uid,v in prior['observed_recovery'][field].items()})
 rec.failed={int(uid):[(np.array(t),np.array(p)) for t,p in values] for uid,values in prior['observed_recovery']['failed'].items()}
 g=f['observed_graphs_diagnostic_only']['21'];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
 for a,b in g['zero_adjacency_edges']:adj[a,b]=0
 agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
 own=f['selected_own_observations']['21'];robot=types.SimpleNamespace(robot_id=21,position=np.array(own['position']),heading=own['heading']);runtime.robots=[robot]
 sched=cls.__new__(cls);sched.runtime=runtime;sched._search_recovery=rec;sched.search_reach_radius=3.5;sched.safety=types.SimpleNamespace(hazards=hazards)
 anchor=np.asarray(f['public_relay_assignments']['6']['anchor']);route=observed_next_hop(agent,anchor)
 from integrations.gppo.extended_travel import extended_observed_next_hop
 details={};candidate=extended_observed_next_hop(21,robot.position,anchor,agent,rec,segment_filter=lambda a,b:not hazards.segment_intersects_warning_buffer(a,b,step=480),route_details=details)
 print('ROUTE_DEBUG',kind,candidate,details,rec.prefixes,flush=True)
 action=sched._relay_action(robot,anchor,agent,route,(robot.position.copy(),robot.heading))
 state,_=KinematicSimple().step(robot.position,action[0],robot.heading,action[1],dt=.1);blocked,reason=om.check_collision(state['position'],radius=.2)
 results.append(dict(kind=kind,position=robot.position.tolist(),command=action[0].tolist(),heading=action[1],first_actual_proposal=state['position'].tolist(),observed_FREE=rec.segment_safe(robot.position,action[0]),warning_clear=not hazards.segment_intersects_warning_buffer(robot.position,action[0],step=480),physics_oracle_collision=blocked,collision_reason=reason,visible_peers=[p.tolist() for p in rec.nearby_positions.get(21,[])]))
 if kind=='old':np.testing.assert_allclose(action[0],f['actual_commands']['21']['waypoint'])
assert results[1]['observed_FREE'] and results[1]['warning_clear'] and not results[1]['physics_oracle_collision']
print(json.dumps(results),flush=True)
assert not np.allclose(results[0]['command'],results[1]['command'])
report=dict(results=results,old_exact_command_reproduced=True,scope='Uses prior470 cache plus current480 own sensors/observed graph; upstream MARVEL base action is not captured, but old command reproduces before fallback. True map only checks independent physical first tick. No full dynamic trajectory claim.',passed=True,acceptance_claim=False)
print(json.dumps(report))
with (DIR/'cycle_boundary_replay.json').open('x') as stream:json.dump(report,stream,indent=2)
