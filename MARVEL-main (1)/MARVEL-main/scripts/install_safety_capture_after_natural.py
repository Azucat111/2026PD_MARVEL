from pathlib import Path
import sys,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
folder=ROOT/'artifacts/extended_integrated_candidate_design_20261011_attempt1';natural=ROOT/'artifacts/extended_integrated_natural_61206_30_dynamic_20261011_attempt1'
assert (natural/'comparison.json').exists();assert json.loads((natural/'comparison.json').read_text())['valid']
assert json.loads((folder/'safety_capture_staged_tests_v2.json').read_text())['passed']
dst=ROOT/'scripts/gppo_motion_capture.py';before=dst.read_bytes();saved=folder/'gppo_motion_capture_before.py'
with saved.open('xb') as s:s.write(before)
shutil.copyfile(folder/'gppo_motion_capture.py',dst)
import pytest
code=pytest.main(['platform_tests/test_gppo_motion_capture.py','platform_tests/test_gppo_safety_motion_capture.py','-q','-p','no:cacheprovider'])
with (folder/'safety_capture_actual_tests.json').open('x') as s:json.dump(dict(passed=code==0,exit_code=code,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),diagnostics_only=True,production_execution_unchanged=True),s,indent=2)
sys.exit(code)
