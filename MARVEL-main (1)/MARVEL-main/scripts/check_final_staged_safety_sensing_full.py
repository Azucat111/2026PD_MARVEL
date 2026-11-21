from pathlib import Path
import sys,json,time,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'artifacts/extended_warning_residual_snapshot_20261015_attempt23'
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['utils','integrations'] for p in (ROOT/d).rglob('*.py')}
import integrations.gppo.extended_safety_motion as safety
import integrations.gppo.scheduler as scheduler
import integrations.gppo.search_recovery as recovery
stage_loaded_sha=hashlib.sha256((OUT/'merged_extended_safety_motion.py').read_bytes()).hexdigest()
recovery_loaded_sha=hashlib.sha256((OUT/'staged_search_recovery.py').read_bytes()).hexdigest()
rec_spec=importlib.util.spec_from_file_location('integrations.gppo.final_stage_recovery',OUT/'staged_search_recovery.py');rec_stage=importlib.util.module_from_spec(rec_spec);rec_spec.loader.exec_module(rec_stage);recovery.ObservedSearchRecovery=rec_stage.ObservedSearchRecovery
spec=importlib.util.spec_from_file_location('integrations.gppo.final_stage_sensing',OUT/'merged_extended_safety_motion.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
safety.ObservedMotionSafetyLayer=stage.ObservedMotionSafetyLayer;scheduler.ObservedMotionSafetyLayer=stage.ObservedMotionSafetyLayer
import pytest
files=[str(p) for p in (ROOT/'platform_tests').glob('test_*.py') if p.name not in ['test_cuda_policy_parity.py','test_dijkstra_parity.py']]
started=time.perf_counter();code=pytest.main(['-q','-p','no:cacheprovider',*files,str(OUT/'test_safety_sensing.py')])
stage_stable=stage_loaded_sha==hashlib.sha256((OUT/'merged_extended_safety_motion.py').read_bytes()).hexdigest() and recovery_loaded_sha==hashlib.sha256((OUT/'staged_search_recovery.py').read_bytes()).hexdigest()
stable=stage_stable and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
r=dict(passed=code==0 and stable,exit_code=int(code),seconds=time.perf_counter()-started,production_source_stable=stable,stage_sha256=hashlib.sha256((OUT/'staged_safety_sensing.py').read_bytes()).hexdigest(),scope='Platform suite excluding unchanged CUDA/Dijkstra parity; plus real sensing recovery tests',acceptance_claim=False)
(OUT/'final_staged_full_regression.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);assert r['passed']
