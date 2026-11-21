"""Full Search teacher; bounded physical preflight only."""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
import numpy as np
from gppo_benchmark_domains import load_domain
from route_visibility_teacher import RouteVisibilityTeacher
from integrations.gppo.opportunity_features import opportunity_batch,public_search_context
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--uavs',type=int,required=True);p.add_argument('--dynamic',action='store_true');p.add_argument('--mode',choices=['heat'],required=True);a=p.parse_args()
OUT=ROOT/'artifacts/extended_heat_facing_probe_20261015_attempt32';plan=json.loads((OUT/'plan.json').read_text());assert a.seed in plan['training_seeds']
for n,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
out=OUT/'data'/f'seed_{a.seed}_{a.mode}';out.mkdir(parents=True,exist_ok=False)
loader=bench.load_and_validate_scenario;bench.load_and_validate_scenario=lambda f:load_domain(loader,f,'configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml',a.dynamic,.2)
WARM=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth';warm_sha=hashlib.sha256(WARM.read_bytes()).hexdigest()
base_runtime=bench.SimulationRuntime;teacher=None
class CaptureRuntime(base_runtime):
 def __init__(self,config):
  global teacher
  config['task_scheduler']['extended_checkpoint']=str(WARM);config['task_scheduler']['extended_checkpoint_sha256']=warm_sha
  super().__init__(config);teacher=RouteVisibilityTeacher(self,'greedy',plan['minimum_gain'])
def choose(graph,deterministic=True):
 before=len(teacher.rows);choice=bench.choose_rule_assignment(graph,deterministic)
 if len(teacher.rows)>before:
  scheduler=teacher.runtime.high_level_scheduler;observer=scheduler.allocator.adapter.heading_observations
  observer.public_search_context=public_search_context(scheduler);batch=opportunity_batch(graph,observer,'cpu')
  teacher.rows[-1].update(actor_batch={k:v.cpu().tolist() for k,v in batch.items()},chosen_action=int(choice.flat_action),uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks])
 return choice
source=(ROOT/'scripts/benchmark_fair_search.py').read_text();source=source[:source.index("if __name__=='__main__':")]
needle="if (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break"
replacement="if detection_efficiency(runtime.events,runtime.search_scenario.target_count,runtime.dt,steps*interval*runtime.dt)['detected_targets']==runtime.search_scenario.target_count or (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break"
namespace=dict(bench.__dict__);exec(compile(source.replace(needle,replacement),str(ROOT/'scripts/benchmark_fair_search.py'),'exec'),namespace)
base_adapter=namespace['MARVELPolicyAdapter']
class PreviewAdapter(base_adapter):
 def setup(self):
  super().setup();original_apply=self.scheduler.apply
  from heat_facing_probe import heat_facing_action
  from utils.dynamics_models import KinematicSimple
  original_search=self.scheduler._search_action
  teacher.heading_changes=[]
  def search(robot,anchor,agent,route,base_action,**kwargs):
   command=original_search(robot,anchor,agent,route,base_action,**kwargs)
   model=teacher.runtime.dynamics_by_type.get(robot.robot_type,teacher.runtime.dynamics)
   if type(model) is not KinematicSimple or kwargs.get('service_scan_heading') is not None:return command
   altered=heat_facing_action(robot,anchor,command,self.scheduler._search_recovery,self.scheduler.search_reach_radius)
   if altered is not command:
    assert np.array_equal(altered[0],command[0])
    if abs((altered[1]-command[1]+180)%360-180)>1e-9:teacher.heading_changes.append(dict(step=int(teacher.runtime.current_step),uid=int(robot.robot_id),old=float(command[1]),new=float(altered[1])))
   return altered
  self.scheduler._search_action=search
  def preview_apply(actions):
   teacher.base_actions=actions
   return original_apply(actions)
  self.scheduler.apply=preview_apply
namespace['MARVELPolicyAdapter']=PreviewAdapter
namespace['SimulationRuntime']=CaptureRuntime;namespace['load_and_validate_scenario']=bench.load_and_validate_scenario;namespace['choose_rule_assignment']=choose
summary=namespace['run'](out,'rule',a.uavs,a.seed,30,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=plan['wall_budget'])
for n,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
(out/'teacher_decisions.json').write_text(json.dumps(teacher.rows,indent=2))
result=dict(seed=a.seed,mode=a.mode,metric=summary['metric'],collision_types=summary['collision_types'],policy_failures=summary['policy_execution_failures'],bounded_window_complete=summary['mission_steps']==30 or summary['metric']['detected_targets']==4,decisions=len(teacher.rows),overrides=sum(r['selected']!=r['baseline'] for r in teacher.rows),acceptance_claim=False,goal_complete=False)
result['heading_changes']=teacher.heading_changes
(out/'complete.json').write_text(json.dumps(result,indent=2));print('TEACHER_COMPLETE',json.dumps(result),flush=True)
