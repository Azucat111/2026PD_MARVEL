from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
old=ast.parse((ROOT/'scripts/run_integrated_training_dev9170_attempt19.py').read_text())
new=ast.parse((ROOT/'scripts/continue_completed_opportunity_attempt19.py').read_text())
# The comparison suffix starts at completed cohort loading, after train invocation.
def suffix(tree):
 i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='folder' for t in n.targets))
 body=tree.body[i:]
 # Explicitly permitted: don't repeat an already successful immutable probability audit.
 for i,n in enumerate(body):
  if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp) and 'rollout_audit_complete256.json' in ast.unparse(n.test):
   assert len(n.body)==1 and not n.orelse
   body[i]=n.body[0]
 return ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)
assert suffix(old)==suffix(new),'Development/final logic changed'
source=(ROOT/'scripts/continue_completed_opportunity_attempt19.py').read_text()
assert "os.environ['PYTHONPATH']=str(ROOT)" in source
assert "Original training must complete; never rerun training" in source
assert 'PREDECLARED_TRAINING_START' not in source
plan=json.loads((ROOT/'artifacts/extended_integrated_training_plan_20261013_attempt19.json').read_text())
assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in {**plan['source_hashes'],**plan['pipeline_hashes']}.items())
watcher=ROOT/'scripts/wait_continue_completed_opportunity_attempt19.py';ast.parse(watcher.read_text())
result=dict(passed=True,post_training_development_final_ast_unchanged=True,all_guarded_sources_unchanged=True,training_rerun=False,continuation_sha256=hashlib.sha256(source.encode()).hexdigest(),watcher_sha256=hashlib.sha256(watcher.read_bytes()).hexdigest(),note='Watcher state goal/reserved flags are startup defaults; final result artifacts are authoritative after the final gate.')
(ROOT/'artifacts/extended_public_opportunity_design_20261013_attempt1/continuation_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
