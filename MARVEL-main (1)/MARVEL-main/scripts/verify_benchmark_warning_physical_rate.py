"""Bounded real-runtime physical unit audit; not an efficiency/Safety claim."""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from gppo_benchmark_domains import load_domain,validate_physical_warning_rate
from utils.scenario_config import load_and_validate_scenario
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.config_profile import prepare_gppo_config
from utils.simulation_runtime import SimulationRuntime
import torch
torch.set_num_threads(1)
checkpoint=Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth')
before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
_,cp=load_frozen_gppo(checkpoint,device='cpu')
cfg=load_domain(load_and_validate_scenario,ROOT/'configs/scenarios/extended_connected_gppo_20261010.yaml',dynamic=True,warning_rate_mps=.2)
cfg['robots']=[dict(id_range=[0,29],type='explorer',team_id=1,config=dict(fov=120,sensor_range=10.,velocity=1.,yaw_rate=35,initial_positions='random_safe'))]
cfg['scenario']['random_seed']=9161
cfg=prepare_gppo_config(cfg,cp);validate_physical_warning_rate(cfg)
np.random.seed(9161);runtime=SimulationRuntime(cfg);runtime.reset();rows=[]
obstacle=runtime.obstacles.dynamic_obstacles[0]
for tick in range(obstacle.spawn_step+21):
    runtime.step([(r.position.copy(),r.heading) for r in runtime.robots])
    if tick in (obstacle.spawn_step,obstacle.spawn_step+10,obstacle.spawn_step+20):
        elapsed=(tick-obstacle.spawn_step)*runtime.dt
        expected=min(obstacle.max_radius,.2*elapsed)
        assert math.isclose(obstacle.radius,expected,rel_tol=1e-9,abs_tol=1e-12)
        rows.append(dict(physics_step=tick,seconds_after_spawn=elapsed,actual_radius_m=obstacle.radius,expected_radius_m=expected))
result=dict(uavs=30,physics_dt=runtime.dt,declared_speed_mps=.2,resolved_rate_per_physics_tick=obstacle.expansion_rate,samples=rows,passed=True,scope='Actual runtime unit conversion only; no Safety or efficiency claim',frozen_checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before)
with (ROOT/'artifacts/extended_dynamic_rate_units_20261010_attempt1/actual_runtime_verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
