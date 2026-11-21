from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));folder=ROOT/'artifacts/extended_candidate_inference_design_20261011_attempt1'
for name in ['heading_features','heading_adapter']:
 spec=importlib.util.spec_from_file_location('integrations.gppo.'+name,folder/(name+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import pytest
code=pytest.main(['platform_tests/test_extended_candidate_inference.py','-k','native_rejects_explicit or heading_snapshot_clock','-q','-p','no:cacheprovider'])
with (folder/'wiring_red_tests.json').open('x') as stream:json.dump(dict(exit_code=code,passed=code==0,artifact_draft_only=True,production_source_unchanged=True,acceptance_claim=False),stream,indent=2)
sys.exit(code)
