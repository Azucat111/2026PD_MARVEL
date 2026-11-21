from pathlib import Path
import sys,importlib.util,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
folder=ROOT/'artifacts/extended_integrated_candidate_design_20261011_attempt1'
spec=importlib.util.spec_from_file_location('scripts.gppo_motion_capture',folder/'gppo_motion_capture.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import pytest
code=pytest.main(['platform_tests/test_gppo_motion_capture.py','platform_tests/test_gppo_safety_motion_capture.py','-q','-p','no:cacheprovider'])
with (folder/'safety_capture_staged_tests.json').open('x') as s:json.dump(dict(exit_code=code,passed=code==0),s)
sys.exit(code)
