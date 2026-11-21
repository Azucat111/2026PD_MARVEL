"""Training-only second Search counterfactuals; production execution is reused."""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
from gppo_benchmark_domains import load_domain
from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.task_graph import TaskType
from integrations.gppo.opportunity_features import opportunity_batch,public_search_context
from integrations.gppo.gppo_model import GPPOActorCritic
from paired_gain_calibration import select_gain_action
import torch
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--uavs',type=int,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--dynamic',action='store_true');a=p.parse_args()
OUT=ROOT/'artifacts/extended_second_search_preflight_20261015_attempt21';plan=json.loads((OUT/'plan.json').read_text())
assert a.seed in plan['training_seeds'] and a.rank in range(4)
for name,digest in plan['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
out=ROOT/'artifacts/extended_warning_residual_snapshot_20261015_attempt23/replay'
if out.exists():
 assert not (out/'complete.json').exists(),'Completed evidence must be reused'
 out=out.with_name(out.name+'_retry_01')
out.mkdir(parents=True,exist_ok=False)
loader=bench.load_and_validate_scenario
bench.load_and_validate_scenario=lambda f:load_domain(loader,f,'configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml',a.dynamic,.2)
FIRST=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/candidate.pth'
assert hashlib.sha256(FIRST.read_bytes()).hexdigest()==plan['first_candidate_sha256']
cp=torch.load(FIRST,map_location='cpu',weights_only=False);model=GPPOActorCritic(edge_feature_dim=9);model.load_state_dict(cp['model']);model.eval()
WARM=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth';warm_sha=hashlib.sha256(WARM.read_bytes()).hexdigest()
base_runtime=bench.SimulationRuntime;current_runtime=None
class CaptureRuntime(base_runtime):
 def __init__(self,config):
  global current_runtime
  config['task_scheduler']['extended_checkpoint']=str(WARM);config['task_scheduler']['extended_checkpoint_sha256']=warm_sha
  assert config['scenario']['duration']['max_steps']==3000
  super().__init__(config);current_runtime=self
bench.SimulationRuntime=CaptureRuntime
# Sampler-only early stop, after all actual sensor detections; physical steps unchanged.
source=(ROOT/'scripts/benchmark_fair_search.py').read_text()
source=source[:source.index("if __name__=='__main__':")]
needle="if (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break"
replacement="if detection_efficiency(runtime.events,runtime.search_scenario.target_count,runtime.dt,steps*interval*runtime.dt)['detected_targets']==runtime.search_scenario.target_count or (stop_on_completion and runtime.tasks.all_complete()) or time.monotonic()-started>wall_budget_seconds:break"
assert source.count(needle)==1
namespace=dict(bench.__dict__);exec(compile(source.replace(needle,replacement),str(ROOT/'scripts/benchmark_fair_search.py'),'exec'),namespace)
namespace['SimulationRuntime']=CaptureRuntime
namespace['load_and_validate_scenario']=bench.load_and_validate_scenario
teacher=bench.choose_rule_assignment;record=[]
def choice(graph,deterministic=True):
 baseline=teacher(graph,deterministic)
 if baseline is None or len(record)>=2 or graph.subtasks[baseline.task_index].task_type!=TaskType.TARGET_SEARCH:return baseline
 options=[]
 for ti,row in enumerate(graph.action_mask):
  for ui,masked in enumerate(row):
   if not masked:
    task=graph.subtasks[ti];uav=graph.uav_states[ui]
    options.append((-float(task.priority),float(graph.edge_features[ti,ui,2]),int(task.subtask_id),int(uav.uav_id),ti,ui))
 options.sort();assert options[0][2:4]==(baseline.subtask_id,baseline.uav_id)
 assert len(options)>=4
 scheduler=current_runtime.high_level_scheduler;observer=scheduler.allocator.adapter.heading_observations
 observer.public_search_context=public_search_context(scheduler);batch=opportunity_batch(graph,observer,'cpu')
 actions=[x[4]*graph.num_uavs+x[5] for x in options[:4]]
 if not record:
  with torch.no_grad():scores=model(**batch)['flat_logits'][0].numpy()
  action=select_gain_action(scores,batch['action_mask'][0].numpy(),actions,.05);rank=actions.index(action)
 else:rank=a.rank
 _,cost,tid,uid,ti,ui=options[rank]
 record.append(dict(ordinal=len(record)+1,time=graph.current_time,rank=rank,teacher=dict(task=baseline.subtask_id,uid=baseline.uav_id),selected=dict(task=tid,uid=uid,cost=cost),top_public_options=[dict(task=x[2],uid=x[3],cost=x[1]) for x in options[:4]],actor_batch={k:v.cpu().tolist() for k,v in batch.items()},graph=dict(uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks],mask=graph.action_mask.tolist(),raw_uav_features=graph.uav_features.tolist(),raw_task_features=graph.task_features.tolist(),raw_edge_features=graph.edge_features.tolist())))
 return GPPOAssignment(ti*graph.num_uavs+ui,ti,ui,tid,uid,0.)
namespace['choose_rule_assignment']=choice

from integrations.gppo.extended_safety_motion import ObservedMotionSafetyLayer
original_select=ObservedMotionSafetyLayer.select_action
SNAP=ROOT/'artifacts/extended_warning_residual_snapshot_20261015_attempt23'
SNAP.mkdir(exist_ok=True)
snapshots=[]
def captured_select(self,robot,*,step):
 result=original_select(self,robot,step=step)
 if int(robot.robot_id)==1 and step in [100,140,170,190,200,207,208,209]:
  rec=self.recovery
  model=self.runtime.dynamics_by_type.get(robot.robot_type,self.runtime.dynamics)
  row=dict(step=int(step),robot_id=1,position=robot.position.tolist(),heading=float(robot.heading),velocity=float(robot.velocity),dt=float(self.runtime.dt),model_velocity=float(model.velocity),frame=rec.frame.describe(),cells=[[int(c),int(r),int(v)] for (c,r),v in rec.cells.items()],peers=[p.tolist() for p in rec.nearby_positions.get(1,[])],failed=[dict(target=p.tolist(),origin=o.tolist()) for p,o in rec.failed.get(1,[])],selected_target=result[0].tolist(),selected_heading=float(result[1]),hazards=[dict(id=str(o.obstacle_id),center=o.position.tolist(),spawn_step=int(o.spawn_step),initial_radius=float(o.initial_radius),expansion_rate=float(o.expansion_rate),max_radius=float(o.max_radius)) for o in self.runtime.obstacles.dynamic_obstacles])
  snapshots.append(row)
  (SNAP/'snapshots.json').write_text(json.dumps(snapshots,indent=2))
 return result
ObservedMotionSafetyLayer.select_action=captured_select

summary=namespace['run'](out,'rule',a.uavs,a.seed,plan['collection_horizon'],physical_coverage=True,stop_on_completion=False,wall_budget_seconds=plan['wall_budget'])
assert len(record)==2
(out/'search_decisions.json').write_text(json.dumps(record,indent=2))
result=dict(diagnostic_only=True,acceptance_claim=False,args=vars(a),metric=summary['metric'],collision_types=summary['collision_types'],policy_failures=summary['policy_execution_failures'],simulation_seconds=summary['elapsed_simulation_seconds'],bounded_window_complete=summary['mission_steps']==plan['collection_horizon'] or summary['metric']['detected_targets']==summary['metric']['total_targets'],stopped_after_true_find=summary['metric']['detected_targets']==4,first_candidate_sha256=plan['first_candidate_sha256'])
(out/'complete.json').write_text(json.dumps(result,indent=2));print('SECOND_SCOPE_COMPLETE',json.dumps(result),flush=True)
