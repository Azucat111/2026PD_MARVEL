"""Controlled one-UAV real-runtime regression of captured UID54 initial pose."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_safety_motion import ObservedMotionSafetyLayer
out=ROOT/'artifacts/extended_safety_physical_9155_pose54_20261010_attempt2';out.mkdir(exist_ok=False)
config=load_and_validate_scenario(ROOT/'configs/scenarios/extended_live_gppo.yaml')
config.pop('task_scheduler',None);config['tasks']=[]
config['robots']=[dict(id_range=[54,54],type='explorer',config=dict(initial_positions=[[21.659563908084515,20.420314348201913]],velocity=1.,yaw_rate=35.,fov=120.,sensor_range=10.))]
config['dynamic_obstacles']=[dict(id='natural_expanding_warning',type='expanding_circle',spawn_step=1,params=dict(position=[20.,20.],initial_radius=0.,expansion_rate=.15,max_radius=4.))]
runtime=SimulationRuntime(config);runtime.reset();runtime.robots[0].heading=348.32082840674957
obs=runtime._get_observations();recovery=ObservedSearchRecovery(runtime.obstacles.frame);recovery.observe(obs)
safety=ObservedMotionSafetyLayer(runtime,recovery=recovery)
class Hook:
    def assignment_for_uav(self,uid):return None
    def preempt_uav(self,uid):pass
    def before_physics_tick(self,actions):
        safety.pre_motion_update(self,runtime.current_step)
        if 54 in safety.active_uav_ids:return [safety.select_action(runtime.robots[0],step=runtime.current_step)]
        return actions
    def post_physics_step(self):
        safety.post_motion_update(self,runtime.current_step);return []
hook=Hook();runtime.high_level_scheduler=hook
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
    [ROOT/'integrations/gppo/extended_safety_motion.py',ROOT/'utils/simulation_runtime.py',ROOT/'integrations/gppo/search_recovery.py',ROOT/'scripts/verify_extended_safety_physical.py']}
for rel in sources:
    dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
rows=[];collisions=[]
for tick in range(200):
    obs,info=runtime.step([(np.array([16.,20.]),350.)]);recovery.observe(obs);collisions.extend(info['collisions'])
    robot=runtime.robots[0]
    rows.append(dict(step=runtime.current_step,position=robot.position.tolist(),heading=robot.heading,
        feedback=obs[54]['motion_feedback'],safety_active=sorted(safety.active_uav_ids),collisions=info['collisions']))
    if safety.successful_escapes:break
result=dict(scope='controlled one-UAV actual runtime; other fleet traffic absent; not whole-fleet acceptance',
    steps=runtime.current_step,collisions=len(collisions),no_fly_violations=safety.no_fly_violations,
    triggers=len(safety.records),escapes=safety.successful_escapes,source_hashes=sources,
    passed=not collisions and safety.no_fly_violations==0 and safety.successful_escapes==1,acceptance_claim=False)
(out/'ticks.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in sources.items())
if not result['passed']:raise RuntimeError('Safety physical regression failed')
