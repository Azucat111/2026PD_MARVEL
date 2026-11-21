from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pytest
folder=ROOT/'artifacts/extended_search_observed_frontier_design_20261011_attempt1'
files=json.loads((ROOT/'artifacts/extended_search_goal_region_20261010_attempt1/regressions.json').read_text())['files']+['platform_tests/test_extended_search_viewpoints.py','platform_tests/test_extended_candidate_inference.py','platform_tests/test_extended_safety_release_route.py','platform_tests/test_extended_search_observed_frontier.py']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for parent in ['utils','integrations'] for p in (ROOT/parent).rglob('*.py')}
code=pytest.main(files+['-q','-p','no:cacheprovider']);assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
with (folder/'actual_production_tests.json').open('x') as s:json.dump(dict(exit_code=code,passed=code==0,production_source_stable=True,source_hashes=hashes,files=files,acceptance_claim=False),s,indent=2)
sys.exit(code)
