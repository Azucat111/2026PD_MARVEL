"""Actual installed recovery/native regressions after staged full suite succeeds."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));STAGE=ROOT/'artifacts/extended_relay_known_route_design_20261013_attempt1'
assert json.loads((STAGE/'staged_full_platform.json').read_text())['passed']
assert json.loads((STAGE/'staged_scheduler_physical.json').read_text())['passed']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['utils','integrations'] for p in (ROOT/d).rglob('*.py')}
import pytest
start=time.perf_counter();files=[str(p) for p in (ROOT/'platform_tests').glob('test_*.py') if p.name not in ['test_cuda_policy_parity.py','test_dijkstra_parity.py']]
code=pytest.main(['-q',*files]);stable=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(passed=code==0 and stable,exit_code=int(code),seconds=time.perf_counter()-start,source_hashes=hashes,source_stable=stable,native_checkpoint_sha256=hashlib.sha256((ROOT/'load_model/MARVEL/checkpoint.pth').read_bytes()).hexdigest(),frozen_gppo_sha256=hashlib.sha256(Path('/home/nick/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth').read_bytes()).hexdigest(),staged_full_suite='staged_full_platform.json',acceptance_claim=False)
(STAGE/'installed_actual_regressions.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}));assert result['passed']
