"""Stop at the first Search allocation; record only existing public preview scores."""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
from gppo_benchmark_domains import load_domain
from route_visibility_teacher import RouteVisibilityTeacher
from integrations.gppo.task_graph import TaskType
p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);p.add_argument('--uavs',required=True,type=int);p.add_argument('--dynamic',action='store_true');a=p.parse_args()
OUT=ROOT/'artifacts/extended_route_visibility_fresh_probe_20261015_attempt31'
plan=json.loads((OUT/'plan.json').read_text());assert a.seed in plan['training_seeds']
for n,h in plan['source_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
out=OUT/'prefix_diagnostics'/str(a.seed);out.mkdir(parents=True,exist_ok=False)
loader=bench.load_and_validate_scenario;bench.load_and_validate_scenario=lambda f:load_domain(loader,f,'configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml',a.dynamic,.2)
base_runtime=bench.SimulationRuntime;teacher=None
class LoggedTeacher(RouteVisibilityTeacher):
 def preview_fraction(self,graph,ti,ui,prior):
  v=super().preview_fraction(graph,ti,ui,prior)
  self.scores.append(dict(task=int(graph.subtasks[ti].subtask_id),uid=int(graph.uav_states[ui].uav_id),fraction=float(v),observed_travel_seconds=float(graph.edge_features[ti,ui,2])))
  return v
class Runtime(base_runtime):
 def __init__(self,config):
  global teacher
  warm=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth'
  config['task_scheduler']['extended_checkpoint']=str(warm);config['task_scheduler']['extended_checkpoint_sha256']=hashlib.sha256(warm.read_bytes()).hexdigest()
  super().__init__(config);teacher=LoggedTeacher(self,'greedy',plan['minimum_gain']);teacher.scores=[]
class Done(Exception):pass
def choose(graph,deterministic=True):
 choice=teacher.choose(graph,deterministic)
 if choice is not None and graph.subtasks[choice.task_index].task_type==TaskType.TARGET_SEARCH:
  row=teacher.rows[-1] if teacher.rows else None
  scores=sorted(teacher.scores,key=lambda r:(-r['fraction'],r['observed_travel_seconds'],r['task'],r['uid']))
  result=dict(seed=a.seed,step=int(teacher.runtime.current_step),choice=vars(choice),decision=row,scores=scores,minimum_gain=plan['minimum_gain'],actual_production_execution_unchanged=True,acceptance_claim=False)
  (out/'complete.json').write_text(json.dumps(result,indent=2));print('PREFIX_CAPTURED',a.seed,'step',result['step'],'scores',len(scores),'best',scores[:1],flush=True)
  raise Done
 return choice
base_adapter=bench.MARVELPolicyAdapter
class Adapter(base_adapter):
 def setup(self):
  super().setup();original=self.scheduler.apply
  def apply(actions):
   teacher.base_actions=actions
   return original(actions)
  self.scheduler.apply=apply
bench.MARVELPolicyAdapter=Adapter;bench.SimulationRuntime=Runtime;bench.choose_rule_assignment=choose
try:bench.run(out,'rule',a.uavs,a.seed,30,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=900)
except Done:pass
assert (out/'complete.json').exists()
