"""Observed-only replay of captured Relay parking and current repair; physics oracle separated."""
from pathlib import Path
import sys,json,types,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.geometry import GeometryFrame
from utils.obstacle_manager import ObstacleManager
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder,mesh_components
from integrations.gppo.observed_graph import observed_next_hop
from utils.dynamics_models import KinematicSimple
PAIR=ROOT/'artifacts/extended_corrected_units_joint_30_dynamic_9162_20261010_attempt1'
OUT=ROOT/'artifacts/extended_dynamic_relay_warning_repair_20261010_attempt1'

def old_class(file,name):
    module=types.ModuleType('integrations.gppo.offline_before_'+name);module.__package__='integrations.gppo';sys.modules[module.__name__]=module
    exec(compile(file.read_text(),str(file),'exec'),module.__dict__)
    return getattr(module,name)

results=[]
for arm in ('gppo','rule'):
    frames=[json.loads(line) for line in (PAIR/arm/'mission_frames.jsonl').read_text().splitlines()]
    f=next(row for row in frames if row['step']==790)
    trace=json.loads((PAIR/arm/'joint_lifecycle_trace.json').read_text());last=trace[-1]
    cfg=json.loads((PAIR/arm/'config.json').read_text());meta=f['observed_recovery']['frame']
    rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
    rec.observe({int(k):v for k,v in f['selected_own_observations'].items()})
    assert rec._authoritative_areal_grid
    rec.cells={(c,r):v for c,r,v in f['observed_recovery']['cells']}
    for field in ('prefixes','prefix_headings','anchors'):
        setattr(rec,field,{int(uid):np.array(v) if field!='prefix_headings' else float(v) for uid,v in f['observed_recovery'][field].items()})
    om=ObstacleManager(cfg['environment'],cfg['dynamic_obstacles']);om.load_from_file(cfg['environment']['map_file']);om.step(790,.1)
    runtime=types.SimpleNamespace(current_step=790,obstacles=om,geometry_mode='extended')
    hazards=SharedHazardAdapter(runtime,warning_margin=cfg['task_scheduler'].get('safety_warning_margin',5.),safe_margin=8.)
    g=f['observed_graphs_diagnostic_only']['12'];coords=np.array(g['node_coords']);adj=np.ones((len(coords),len(coords)))
    for a,b in g['zero_adjacency_edges']:adj[a,b]=0
    agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.array(g['neighbor_indices']))
    robot=types.SimpleNamespace(robot_id=12,position=np.array(f['selected_own_observations']['12']['position']),heading=f['selected_own_observations']['12']['heading'])
    runtime.robots=[robot]
    scheduler=GPPOTaskScheduler.__new__(GPPOTaskScheduler);scheduler.runtime=runtime;scheduler._search_recovery=rec;scheduler.search_reach_radius=3.5
    scheduler.safety=types.SimpleNamespace(hazards=hazards,active_uav_ids=set(),marvel_agent=lambda uid:agent)
    old_scheduler=old_class(OUT/'scheduler.py.before','GPPOTaskScheduler')
    anchor=np.array(f['public_relay_assignments']['2']['anchor']);route=observed_next_hop(agent,anchor)
    before=old_scheduler._search_action(scheduler,robot,anchor,agent,route,(robot.position.copy(),robot.heading))
    final=old_scheduler._apply_hazard_traversability(scheduler,[before],{12:0})[0]
    np.testing.assert_allclose(final[0],f['actual_commands']['12']['waypoint'])
    positions={int(uid):np.array(p) for uid,p in last['public_delayed_positions'].items()}
    assignment={int(t):(int(a['helper']),np.array(a['anchor'])) for t,a in last['relay_assignments'].items()}
    provider=scheduler._observed_relay_anchor_candidates
    old_builder=old_class(OUT/'mesh_relay.py.before','ObservedMeshRelayDemandBuilder')(comm_range=20,max_hops=2,max_demands=2)
    old_builder.anchor_candidates_provider=provider;old_a={k:(u,p.copy()) for k,(u,p) in assignment.items()};old_builder.refresh_assignments(old_a,positions)
    builder=ObservedMeshRelayDemandBuilder(comm_range=20,max_hops=2,max_demands=2);builder.anchor_candidates_provider=provider
    new_a={k:(u,p.copy()) for k,(u,p) in assignment.items()};print("PROVIDER_BEST",sorted(zip(builder._candidate_lcc_sizes(np.asarray(provider()),positions,12,new_a).tolist(),np.asarray(provider()).tolist()),reverse=True)[:12],flush=True); print("CANDIDATE_DEBUG",builder._candidate_lcc_sizes(np.asarray([[33.,26.],[21.5,10.5],[16.5,47.5]]),positions,12,new_a),flush=True); builder.refresh_assignments(new_a,positions); print("ASSIGNMENTS",new_a,flush=True)
    new_anchor=new_a[2][1]
    action=scheduler._search_action(robot,new_anchor,agent,observed_next_hop(agent,new_anchor),(robot.position.copy(),robot.heading))
    state,_=KinematicSimple().step(robot.position,action[0],robot.heading,action[1],dt=.1)
    blocked,kind=om.check_collision(state['position'],radius=.2)
    forecast=dict(positions)
    for helper,point in new_a.values():forecast[helper]=point
    groups=mesh_components(positions,20,exclude_uav_ids=[u for u,_ in assignment.values()])
    result=dict(arm=arm,step=790,own_sensor_position=robot.position.tolist(),old_cached_prefix=before[0].tolist(),captured_final_command=final[0].tolist(),old_prefix_warning_blocked=hazards.segment_intersects_warning_buffer(robot.position,before[0],step=790),old_anchor_warning_blocked=hazards.segment_intersects_warning_buffer(anchor,anchor,step=790),public_component_sizes=[len(x) for x in groups],old_retarget_anchor=old_a[2][1].tolist(),new_retarget_anchor=new_anchor.tolist(),new_first_command=action[0].tolist(),new_prefix_observed_free=rec.segment_safe(robot.position,action[0]),new_prefix_warning_clear=not hazards.segment_intersects_warning_buffer(robot.position,action[0],step=790),actual_supported_first_tick=state['position'].tolist(),physics_oracle_collision=blocked,collision_kind=kind,radio_forecast_lcc=max(map(len,mesh_components(forecast,20)))/len(forecast),radio_forecast_only_not_physical_arrival=True,missing_state=['historical sensing_uids set is not captured; old final command still exactly reproduced'],scope='Observed labels, own sensors, delayed public radio poses and public warning select the repair; true map is used only for an independent first-tick physical collision check')
    print(json.dumps(result),flush=True)
    assert result['old_prefix_warning_blocked'] and result['old_anchor_warning_blocked']
    assert not np.array_equal(new_anchor,anchor) and result['new_prefix_observed_free'] and result['new_prefix_warning_clear']
    assert np.linalg.norm(state['position']-robot.position)>0 and not blocked and result['radio_forecast_lcc']==1.
    results.append(result)
with (OUT/'captured_reconstruction.json').open('x') as stream:json.dump(dict(results=results,passed=True,acceptance_claim=False),stream,indent=2)
print(json.dumps(results))
