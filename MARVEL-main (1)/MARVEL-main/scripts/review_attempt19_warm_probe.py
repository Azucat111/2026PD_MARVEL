"""Bounded pre-attempt19-policy diagnostic, not final acceptance."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_fair_search as bench
from gppo_benchmark_domains import load_domain
candidate=ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/initial_opportunity.pth'
sha=hashlib.sha256(candidate.read_bytes()).hexdigest();original=bench.SimulationRuntime;base_loader=bench.load_and_validate_scenario
class WarmRuntime(original):
 def __init__(self,config):
  config['task_scheduler']['extended_checkpoint']=str(candidate)
  config['task_scheduler']['extended_checkpoint_sha256']=sha
  super().__init__(config)
bench.SimulationRuntime=WarmRuntime
bench.load_and_validate_scenario=lambda f:load_domain(base_loader,f,'configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml',False,.2)
out=ROOT/'artifacts/extended_system_review_20261014_attempt19/warm_9170_30_static_H20';out.mkdir(exist_ok=False)
(out/'manifest.json').write_text(json.dumps(dict(diagnostic_only=True,acceptance_claim=False,seed=9170,horizon_seconds=20,checkpoint_sha256=sha,comparison='pre-attempt19 warm policy, same declared map/runtime/masks/seed as completed development; prior H300 runs retained'),indent=2))
summary=bench.run(out,'gppo',30,9170,20,physical_coverage=True,stop_on_completion=False,wall_budget_seconds=180)
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==sha
print('WARM_DIAGNOSTIC',json.dumps(summary['metric']))
