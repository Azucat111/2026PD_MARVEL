"""Controlled initial Relay reservation -> real Safety preemption/escape.
This is a protocol fixture, not natural mission or efficiency acceptance.
All subsequent motion, sensors, collisions and controllers are production.
"""
from pathlib import Path
import sys,json,time,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime
from utils.policy_adapter import MARVELPolicyAdapter
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.config_profile import prepare_gppo_config
CHECKPOINT=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
OUT=ROOT/'artifacts/dynamic_safety_lifecycle_20261010_attempt3'
OUT.mkdir(exist_ok=False)
grid=np.zeros((80,80),dtype=np.uint8);grid[:4]=1;grid[-4:]=1;grid[:,:4]=1;grid[:,-4:]=1
# Occupancy .npy uses the actual loader convention FREE=255/OCCUPIED=1.
np.save(OUT/'fixture.npy',np.where(grid==1,1,255).astype(np.uint8))
results=[]
for uavs in (30,60):
    config=load_and_validate_scenario(ROOT/'configs/scenarios/extended_live_gppo.yaml')
    config['environment']['map_file']=str(OUT/'fixture.npy')
    positions=[[8.,8.]]+[[24.+4*(i%8),8.+4*(i//8)] for i in range(uavs-1)]
    config['robots']=[{'id_range':[0,uavs-1],'type':'explorer','config':{
        'fov':120,'sensor_range':10.,'velocity':1.,'yaw_rate':35,'initial_positions':positions}}]
    config['dynamic_obstacles']=[{'id':'public_warning_fixture','type':'expanding_circle','spawn_step':1,
        'params':{'position':[5.,8.],'initial_radius':.5,'expansion_rate':0.,'max_radius':.5}}]
    config['task_scheduler']={'mode':'gppo','checkpoint':str(CHECKPOINT),'base_position':[36.,24.],'search_scenario_seed':12345}
    _,checkpoint=load_frozen_gppo(str(CHECKPOINT),device='cpu');config=prepare_gppo_config(config,checkpoint)
    config['policy']={'device':'cuda','batch_inference':True}
    np.random.seed(12345)
    runtime=SimulationRuntime(config);runtime.reset()
    for robot in runtime.robots:robot.heading=0.
    adapter=MARVELPolicyAdapter(runtime);adapter.setup();scheduler=adapter.scheduler
    obs=runtime._get_observations()
    topology=runtime.comm.get_topology([r.position for r in runtime.robots])
    connected,_=runtime.comm.check_connectivity(topology)
    for event in runtime.tasks.update(0,runtime.robots,obs,
            exploration_rate=runtime.exploration_rate,comm_connected=connected,
            connectivity_ratio=runtime.comm.connectivity_ratio(topology),
            target_detections=runtime._get_target_detections(obs)):
        runtime._log_event(event['type'],event)
    (OUT/f'config_{uavs}.json').write_text(json.dumps(config,indent=2,default=str))
    # Declared initial public reservation. No hidden target/map decision.
    scheduler.relay_assignments={1:(0,np.array([12.,8.]))}
    scheduler._apply_uav_state(0,'relay',6.)
    obs=runtime._get_observations();rows=[];collisions=[]
    for mission in range(1,13):
        actions=adapter.get_actions(obs)
        for _ in range(runtime._mission_step_interval()):
            obs,info=runtime.step(actions);collisions.extend(info['collisions'])
        risk=scheduler.safety.hazards.risk_at(runtime.robots[0].position)
        rows.append({'mission':mission,'uid0_position':runtime.robots[0].position.tolist(),
            'uid0_role':runtime.robots[0].current_task,'safety_active':sorted(scheduler.safety.active_uav_ids),
            'clearance':None if risk is None else risk.clearance,'records':len(scheduler.safety.records),
            'escapes':scheduler.safety.successful_escapes,'search':dict(scheduler.search_assignments)})
    summary={'uavs':uavs,'controlled_fixture':True,'natural_from_reset':False,
        'preempted_relay':any(r.uav_id==0 and r.preempted_assignment==('relay',1) for r in scheduler.safety.records),
        'released_after_real_escape':any(r.uav_id==0 and r.escaped_step is not None for r in scheduler.safety.records),
        'records':[r.__dict__ for r in scheduler.safety.records],
        'uid0_travel_m':runtime.robots[0].travel_distance,'collisions':collisions,
        'no_fly_violations':scheduler.safety.no_fly_violations,'final_active':sorted(scheduler.safety.active_uav_ids)}
    (OUT/f'case_{uavs}.json').write_text(json.dumps({'summary':summary,'trajectory':rows,'events':runtime.events},indent=2,default=str))
    results.append(summary);print(json.dumps(summary),flush=True)
(OUT/'summary.json').write_text(json.dumps(results,indent=2))
assert all(r['preempted_relay'] and r['released_after_real_escape'] and not r['collisions'] and not r['no_fly_violations'] for r in results)
