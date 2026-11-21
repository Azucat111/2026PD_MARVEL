from pathlib import Path
import sys,importlib.util,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PATH=ROOT/'artifacts/extended_public_peripheral_search_design_20261011_attempt1/search_viewpoints.py'
spec=importlib.util.spec_from_file_location('integrations.gppo.search_viewpoints',PATH);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
scheduler_path=PATH.parent/'scheduler.py'
spec2=importlib.util.spec_from_file_location('integrations.gppo.scheduler',scheduler_path);module2=importlib.util.module_from_spec(spec2);sys.modules[spec2.name]=module2;spec2.loader.exec_module(module2)
import integrations.gppo
integrations.gppo.GPPOTaskScheduler=module2.GPPOTaskScheduler
import pytest
files=json.loads((ROOT/'artifacts/extended_search_goal_region_20261010_attempt1/regressions.json').read_text())['files']+['platform_tests/test_extended_search_viewpoints.py']
code=pytest.main(files+['-q','-p','no:cacheprovider'])
result=dict(exit_code=code,passed=code==0,artifact_draft_only=True,production_source_unchanged=True,acceptance_claim=False)
with (PATH.parent/'scheduler_staged_regressions.json').open('x') as stream:json.dump(result,stream,indent=2)
sys.exit(code)
