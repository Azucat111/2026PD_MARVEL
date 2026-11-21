"""Bounded real extended mesh Relay bridging and counterfactual release fixture.
No Search prior; nonparticipants hold. Targets later fly a declared public
return command, without teleportation. This is not natural efficiency evidence.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gppo_motion_capture import capture_motion_frame,append_motion_frame
import torch
torch.set_num_threads(1)
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime
from utils.policy_adapter import MARVELPolicyAdapter
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.config_profile import prepare_gppo_config
CHECKPOINT=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
OUT=ROOT/'artifacts/mesh_relay_physical_lifecycle_20261010_attempt6';OUT.mkdir(exist_ok=False)
np.save(OUT/'fixture.npy',np.pad(np.full((72,72),255,dtype=np.uint8),4,constant_values=1))

source_paths=list((ROOT/'integrations').rglob('*.py'))+list((ROOT/'utils').rglob('*.py'))+[ROOT/'scripts/gppo_motion_capture.py',ROOT/'scripts/verify_observed_mesh_relay_physical.py',ROOT/'configs/scenarios/extended_live_gppo.yaml']
source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
frozen_hash=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
for p in source_paths:
    saved=OUT/'source_snapshot'/p.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(p.read_bytes())
(OUT/'manifest.json').write_text(json.dumps(dict(source_hashes=source_hashes,frozen_checkpoint_sha256=frozen_hash,controlled_fixture=True,acceptance_claim=False),indent=2))
summary=[]
for uavs,count in ((30,4),(60,7)):
    config=load_and_validate_scenario(ROOT/'configs/scenarios/extended_live_gppo.yaml')
    config['environment']['map_file']=str(OUT/'fixture.npy')
    targets=[[52.,12.+4*i] for i in range(count)]
    cores=[[8.+4*(i%6),8.+4*(i//6)] for i in range(uavs-count)]
    cfg={'fov':120,'sensor_range':10.,'velocity':1.,'yaw_rate':35}
    config['robots']=[{'id_range':[0,count-1],'type':'explorer','config':{**cfg,'initial_positions':targets}},
        {'id_range':[count,uavs-1],'type':'relay','config':{**cfg,'initial_positions':cores}}]
    for task in config['tasks']:
        if task['type']=='relay':task['assigned_robot_types']=['relay']
        elif task['type']=='exploration':task['assigned_robot_types']=['explorer','relay']
    config['task_scheduler']={'mode':'gppo','checkpoint':str(CHECKPOINT),'base_position':[20.,24.],'search_scenario_seed':12345}
    _,checkpoint=load_frozen_gppo(str(CHECKPOINT),device='cpu');config=prepare_gppo_config(config,checkpoint)
    config['policy']={'device':'cuda','batch_inference':True}
    runtime=SimulationRuntime(config);runtime.reset()
    for robot in runtime.robots:robot.heading=0.
    adapter=MARVELPolicyAdapter(runtime);adapter.setup();scheduler=adapter.scheduler
    scheduler.install_search_scenario([]) # Declared Relay-only public fixture.
    obs=runtime._get_observations();rows=[];collisions=[]
    initial=scheduler.relay_builder.snapshot({r.robot_id:r.position for r in runtime.robots},base_position=scheduler.base_position)
    assert not set(range(count))&set(initial.connected_uav_ids)
    bridged=False;indispensable=False;bridge_step=None;moving_targets=False
    started=time.monotonic()
    for mission in range(1,61):
        actions=adapter.get_actions(obs)
        append_motion_frame(OUT/f'case_{uavs}_mission_frames.jsonl',capture_motion_frame(runtime,scheduler,obs,actions))
        helpers={uid for uid,_ in scheduler.relay_assignments.values()}
        for index,robot in enumerate(runtime.robots):
            if robot.robot_id in helpers or robot.robot_id in scheduler.safety.active_uav_ids:continue
            if moving_targets and robot.robot_id<count:
                goal=np.array([28.,12.+4*robot.robot_id]);delta=goal-robot.position
                actions[index]=(goal,float(np.degrees(np.arctan2(delta[1],delta[0]))%360.))
            else:actions[index]=(robot.position.copy(),robot.heading)
        for _ in range(runtime._mission_step_interval()):
            obs,info=runtime.step(actions);collisions.extend(info['collisions'])
        positions={r.robot_id:r.position for r in runtime.robots}
        snapshot=scheduler.relay_builder.snapshot(positions,base_position=scheduler.base_position)
        counter=scheduler.relay_builder.snapshot(positions,base_position=scheduler.base_position,exclude_uav_ids=helpers)
        connected=set(range(count))<=set(snapshot.connected_uav_ids)
        if connected and helpers and not set(range(count))<=set(counter.connected_uav_ids):
            bridged=True;indispensable=True;bridge_step=mission;moving_targets=True
        rows.append({'mission':mission,'target_positions':[runtime.robots[i].position.tolist() for i in range(count)],
            'helper_positions':{uid:runtime.robots[uid].position.tolist() for uid in helpers},
            'assignments':{target:uid for target,(uid,_) in scheduler.relay_assignments.items()},
            'connected_ratio':snapshot.connectivity_ratio,'counterfactual_ratio':counter.connectivity_ratio,
            'targets_connected':connected,'counterfactual_targets_connected':set(range(count))<=set(counter.connected_uav_ids),
            'release_count':scheduler.relay_release_gate.release_count})
        (OUT/f'case_{uavs}_progress.json').write_text(json.dumps(rows,indent=2))
        if mission%10==0:print('RELAY',uavs,mission,rows[-1],round(time.monotonic()-started,1),flush=True)
        if bridged and scheduler.relay_release_gate.release_count>0 and not scheduler.relay_assignments:break
        if time.monotonic()-started>300:break
    result={'uavs':uavs,'controlled_fixture':True,'natural_acceptance':False,'radio_semantics':'largest UAV connected component, same as runtime task','initial_connectivity':initial.connectivity_ratio,
        'real_indispensable_bridge':indispensable,'bridge_step':bridge_step,
        'release_count':scheduler.relay_release_gate.release_count,'steps':mission,
        'targets_actual_travel_m':[runtime.robots[i].travel_distance for i in range(count)],
        'collisions':collisions,'final_assignments':{target:uid for target,(uid,_) in scheduler.relay_assignments.items()}}
    (OUT/f'case_{uavs}.json').write_text(json.dumps({'summary':result,'trajectory':rows,'events':runtime.events,'config':config},indent=2,default=str))
    summary.append(result);print(json.dumps(result),flush=True)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
assert all(r['real_indispensable_bridge'] and r['release_count'] and not r['collisions'] and not r['final_assignments'] for r in summary)

assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in source_hashes.items())
assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==frozen_hash
