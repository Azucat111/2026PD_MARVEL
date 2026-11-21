from pathlib import Path
import sys,json,time,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));STAGE=ROOT/'artifacts/extended_paired_calibration_20261014_attempt20/performance_stage'
assert json.loads((STAGE/'physical_parity_cost_v2.json').read_text())['passed']
source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['utils','integrations'] for p in (ROOT/d).rglob('*.py')}
import utils.dynamics_models as dynamics
import integrations.gppo.extended_safety_motion as safety
import integrations.gppo.scheduler as scheduler
spec=importlib.util.spec_from_file_location('utils.full_stage_dynamics',STAGE/'dynamics_models.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
spec=importlib.util.spec_from_file_location('integrations.gppo.full_stage_safety',STAGE/'extended_safety_motion.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
dynamics.KinematicSimple=D.KinematicSimple;safety.ObservedMotionSafetyLayer=S.ObservedMotionSafetyLayer
if hasattr(scheduler,'ObservedMotionSafetyLayer'):scheduler.ObservedMotionSafetyLayer=S.ObservedMotionSafetyLayer
import pytest
started=time.perf_counter();files=[str(p) for p in (ROOT/'platform_tests').glob('test_*.py') if p.name not in ['test_cuda_policy_parity.py','test_dijkstra_parity.py']]
code=pytest.main(['-q','-p','no:cacheprovider',*files,str(STAGE/'test_equivalent_safety_cost.py')])
stable=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
result=dict(passed=code==0 and stable,exit_code=int(code),seconds=time.perf_counter()-started,production_source_stable=stable,scope='Platform suite excluding unchanged CUDA/Dijkstra parity, plus equivalent computation tests',source_hashes=source_hashes,acceptance_claim=False)
(STAGE/'full_platform_regression.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}),flush=True);assert result['passed']
