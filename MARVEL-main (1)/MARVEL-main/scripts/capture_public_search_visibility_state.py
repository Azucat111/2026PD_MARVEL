"""Single-decision counterfactual diagnostic using existing execution primitives."""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
from gppo_benchmark_domains import load_domain
from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.task_graph import TaskType
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--uavs',type=int,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--dynamic',action='store_true');a=p.parse_args()
assert 63400 <= a.seed <= 63407 and a.rank==0
out=ROOT/f'artifacts/extended_public_visibility_preflight_20261015_attempt24/data/seed_{a.seed}'
out.mkdir(parents=True,exist_ok=False)
loader=bench.load_and_validate_scenario
bench.load_and_validate_scenario=lambda f:load_domain(loader,f,'configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml',a.dynamic,.2)
from integrations.gppo.opportunity_features import opportunity_batch,public_search_context
WARM=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth'
WARM_SHA=hashlib.sha256(WARM.read_bytes()).hexdigest()
base_runtime=bench.SimulationRuntime;current_runtime=None
class CaptureRuntime(base_runtime):
 def __init__(self,config):
  global current_runtime
  config['task_scheduler']['extended_checkpoint']=str(WARM)
  config['task_scheduler']['extended_checkpoint_sha256']=WARM_SHA
  super().__init__(config);current_runtime=self
bench.SimulationRuntime=CaptureRuntime
class PublicStateCaptured(Exception):pass
teacher=bench.choose_rule_assignment;record=[]
def choice(graph,deterministic=True):
 baseline=teacher(graph,deterministic)
 if baseline is None or record or graph.subtasks[baseline.task_index].task_type!=TaskType.TARGET_SEARCH:return baseline
 options=[]
 for ti,row in enumerate(graph.action_mask):
  for ui,masked in enumerate(row):
   if not masked:
    task=graph.subtasks[ti];uav=graph.uav_states[ui]
    options.append((-float(task.priority),float(graph.edge_features[ti,ui,2]),int(task.subtask_id),int(uav.uav_id),ti,ui))
 options.sort();assert options[0][2:4]==(baseline.subtask_id,baseline.uav_id)
 assert len(options)>a.rank
 selected=options[a.rank];_,cost,tid,uid,ti,ui=selected
 scheduler=current_runtime.high_level_scheduler
 observer=scheduler.allocator.adapter.heading_observations
 observer.public_search_context=public_search_context(scheduler)
 actor_batch={k:v.cpu().tolist() for k,v in opportunity_batch(graph,observer,'cpu').items()}
 record.append(dict(actor_batch=actor_batch,warm_sha256=WARM_SHA,time=graph.current_time,rank=a.rank,teacher=dict(task=baseline.subtask_id,uid=baseline.uav_id),selected=dict(task=tid,uid=uid,cost=cost),top_public_options=[dict(task=x[2],uid=x[3],cost=x[1]) for x in options[:4]],graph=dict(uav_ids=[u.uav_id for u in graph.uav_states],task_ids=[t.subtask_id for t in graph.subtasks],mask=graph.action_mask.tolist(),raw_uav_features=graph.uav_features.tolist(),raw_task_features=graph.task_features.tolist(),raw_edge_features=graph.edge_features.tolist())))
 rec=scheduler._search_recovery;views=scheduler._search_viewpoints
 metadata={int(r.robot_id):r.robot_type for r in current_runtime.robots}
 public_uavs=[]
 for u in graph.uav_states:
  sensor=current_runtime.sensor_by_type.get(metadata[int(u.uav_id)],current_runtime.sensor)
  dynamics=current_runtime.dynamics_by_type.get(metadata[int(u.uav_id)],current_runtime.dynamics)
  public_uavs.append(dict(uid=int(u.uav_id),position=list(u.position),velocity=float(u.velocity),heading=observer.heading(int(u.uav_id)),fov=float(sensor.fov),sensor_range=float(sensor.sensor_range),yaw_rate=float(dynamics.yaw_rate)))
 record[0]['public_visibility']=dict(frame=rec.frame.describe(),observed_cells=[[int(c),int(r),int(v)] for (c,r),v in rec.cells.items()],uavs=public_uavs,task_positions=[None if s.position is None else list(s.position) for s in graph.subtasks],prior_radius=float(views.prior_radius),checked_cells_by_heat={str(k):list(v) for k,v in views.checked_cells_by_heat.items()},route_estimates=[dict(task=int(k[0]),uid=int(k[1]),details=v) for k,v in scheduler.allocator.last_travel_estimates.items()])
 (out/'public_state.json').write_text(json.dumps(record[0],indent=2,default=lambda x:x.tolist()))
 raise PublicStateCaptured()
bench.choose_rule_assignment=choice
try:
 bench.run(out,'rule',a.uavs,a.seed,3,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=60)
except PublicStateCaptured:
 print('PUBLIC_STATE_CAPTURED',a.seed,flush=True)
assert (out/'public_state.json').exists()
