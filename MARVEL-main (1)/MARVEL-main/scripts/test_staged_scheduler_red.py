from pathlib import Path
import sys,importlib.util,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PATH=ROOT/'artifacts/extended_public_peripheral_search_design_20261011_attempt1/search_viewpoints.py'
spec=importlib.util.spec_from_file_location('integrations.gppo.search_viewpoints',PATH);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import pytest
code=pytest.main(['platform_tests/test_extended_search_viewpoints.py','-k','excluded_service_disk or does_not_exclude or released_search_owner','-q','-p','no:cacheprovider'])
result=dict(exit_code=code,passed=code==0,artifact_draft_only=True,production_source_unchanged=True,acceptance_claim=False)
with (PATH.parent/'scheduler_red_tests.json').open('x') as stream:json.dump(result,stream,indent=2)
sys.exit(code)
