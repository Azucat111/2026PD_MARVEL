"""Full local platform suite with staged sensor module; production files stay fixed."""
from pathlib import Path
import sys,json,hashlib,importlib.util,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));STAGE=ROOT/'artifacts/extended_exact_sensor_cache_design_20261012_attempt1'
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261011_attempt15.json').read_text())
for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
old=ast.parse((STAGE/'sensor_models_before.py').read_text());new=ast.parse((STAGE/'sensor_models_draft.py').read_text())
a=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name=='_visible_cells');b=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='_compute_visible_cells');b.name=a.name;assert ast.dump(a)==ast.dump(b)
import utils
spec=importlib.util.spec_from_file_location('utils.sensor_models',STAGE/'sensor_models_draft.py');draft=importlib.util.module_from_spec(spec);spec.loader.exec_module(draft);sys.modules['utils.sensor_models']=draft;utils.sensor_models=draft
import pytest
class Results:
 def __init__(self):self.passed=[];self.failed=[];self.skipped=[]
 def pytest_runtest_logreport(self,report):
  if report.failed:self.failed.append(report.nodeid+'::'+report.when)
  elif report.skipped:self.skipped.append(report.nodeid)
  elif report.when=='call':self.passed.append(report.nodeid)
result=Results();code=pytest.main(['platform_tests','-q','-p','no:cacheprovider'],plugins=[result])
for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
record=dict(exit_code=code,passed=code==0,tests_passed=len(result.passed),failures=result.failed,skipped=result.skipped,original_visibility_body_ast_identical=True,production_source_stable=True,staged_module_sha256=hashlib.sha256((STAGE/'sensor_models_draft.py').read_bytes()).hexdigest(),production_installed=False,acceptance_claim=False)
(STAGE/'full_platform_staged_regressions.json').write_text(json.dumps(record,indent=2));print(json.dumps(record));sys.exit(code)
