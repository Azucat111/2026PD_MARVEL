"""Canonical staged-module full-platform regression; live sources remain frozen."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));STAGE=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt2'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261012_attempt17.json').read_text());hashes={**plan['source_hashes'],**plan['pipeline_hashes']}
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
for name,file in [('integrations.gppo.relay_frontier','relay_frontier.py'),('integrations.gppo.scheduler','scheduler_draft.py'),('utils.policy_adapter','policy_adapter_draft.py')]:
 spec=importlib.util.spec_from_file_location(name,STAGE/file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 parent,attr=name.rsplit('.',1);setattr(sys.modules[parent],attr,module)
 if name=='integrations.gppo.scheduler':sys.modules['integrations.gppo'].GPPOTaskScheduler=module.GPPOTaskScheduler
 if name=='utils.policy_adapter':module._MARVEL_ROOT=ROOT
import pytest
start=time.perf_counter();code=pytest.main(['-q','platform_tests',str(STAGE/'test_relay_frontier.py'),str(STAGE/'test_relay_probe_integration.py')]);stable=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(passed=code==0,exit_code=int(code),production_source_stable=stable,seconds=time.perf_counter()-start,staged_hashes={p:hashlib.sha256((STAGE/p).read_bytes()).hexdigest() for p in ['relay_frontier.py','scheduler_draft.py','policy_adapter_draft.py']},acceptance_claim=False)
(STAGE/'staged_full_platform_regressions_v2.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert code==0 and stable
