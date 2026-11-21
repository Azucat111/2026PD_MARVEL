from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));folder=ROOT/'artifacts/extended_integrated_candidate_design_20261011_attempt1'
import integrations.gppo
for name in ['heading_features','heading_adapter','event_allocator','search_viewpoints','extended_safety_motion','scheduler']:
 spec=importlib.util.spec_from_file_location('integrations.gppo.'+name,folder/(name+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);setattr(integrations.gppo,name,module)
integrations.gppo.GPPOTaskScheduler=sys.modules['integrations.gppo.scheduler'].GPPOTaskScheduler
import pytest
files=json.loads((ROOT/'artifacts/extended_search_goal_region_20261010_attempt1/regressions.json').read_text())['files']+['platform_tests/test_extended_search_viewpoints.py','platform_tests/test_extended_candidate_inference.py','platform_tests/test_extended_safety_release_route.py']
code=pytest.main(files+['-q','-p','no:cacheprovider'])
with (folder/'broad_safety_tests.json').open('x') as stream:json.dump(dict(exit_code=code,passed=code==0,artifact_draft_only=True,production_source_unchanged=True,acceptance_claim=False),stream,indent=2)
sys.exit(code)
