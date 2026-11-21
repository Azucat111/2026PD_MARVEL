from pathlib import Path
import sys,importlib.util,hashlib,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DRAFT=ROOT/'artifacts/extended_relay_known_route_design_20261013_attempt1'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt18.json').read_text());hashes={**plan['source_hashes'],**plan['pipeline_hashes']}
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
import integrations.gppo as package
for name,file in [('integrations.gppo.relay_frontier','relay_frontier.py'),('integrations.gppo.scheduler','scheduler.py')]:
 spec=importlib.util.spec_from_file_location(name,DRAFT/file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
package.GPPOTaskScheduler=sys.modules['integrations.gppo.scheduler'].GPPOTaskScheduler
import pytest
code=pytest.main(['artifacts/extended_relay_known_route_design_20261013_attempt1/test_known_relay_route.py','platform_tests/test_observed_relay_probe.py','-q','--tb=short'])
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
with (DRAFT/'staged_first_tests.json').open('x') as f:json.dump(dict(passed=code==0,pytest_exit=code,production_unchanged=True,unique_guarded_files=len(hashes),staged_scheduler_sha256=hashlib.sha256((DRAFT/'scheduler.py').read_bytes()).hexdigest(),staged_frontier_sha256=hashlib.sha256((DRAFT/'relay_frontier.py').read_bytes()).hexdigest()),f,indent=2)
raise SystemExit(code)
