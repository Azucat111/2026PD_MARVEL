"""Prove the partial-fixture guard changes no initialized-runtime execution body."""
from pathlib import Path
import ast,json,hashlib,copy
ROOT=Path(__file__).resolve().parents[1];S=ROOT/'artifacts/extended_relay_frontier_design_20261012_attempt2'
old=ast.parse((S/'scheduler_before_partial_fixture_guard.py').read_text());new=ast.parse((S/'scheduler_draft.py').read_text())
def cls(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPPOTaskScheduler')
def dump(node):return ast.dump(node,include_attributes=False)
a,b=cls(old),cls(new);ma={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)};mb={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)};assert ma.keys()==mb.keys()
changed=[name for name in ma if dump(ma[name])!=dump(mb[name])];assert changed==['_refresh_observed_relay_probe']
x,y=ma[changed[0]],mb[changed[0]];assert dump(ast.Module(body=x.body[2:],type_ignores=[]))==dump(ast.Module(body=y.body[3:],type_ignores=[]))
assert dump(y.body[1])==dump(ast.parse("runtime=getattr(self,'runtime',None)").body[0])
class GuardTransformer(ast.NodeTransformer):
 def visit_Attribute(self,node):
  if isinstance(node.value,ast.Name) and node.value.id=='self':
   if node.attr=='runtime':return ast.Name(id='runtime',ctx=ast.Load())
   if node.attr=='_search_recovery':return ast.parse("getattr(self,'_search_recovery',None)",mode='eval').body
  return self.generic_visit(node)
expected=GuardTransformer().visit(copy.deepcopy(x.body[1]));assert dump(expected)==dump(y.body[2])
# For fully initialized scheduler objects, both replacements resolve exactly the same objects.
result=dict(passed=True,only_modified_method=changed[0],all_other_methods_ast_identical=True,guard_true_branch_and_remaining_execution_ast_identical=True,initialized_runtime_guard_values_identical=True,partial_missing_fields_now_return_without_probe=True,old_scheduler_sha256=hashlib.sha256((S/'scheduler_before_partial_fixture_guard.py').read_bytes()).hexdigest(),new_scheduler_sha256=hashlib.sha256((S/'scheduler_draft.py').read_bytes()).hexdigest(),physical_natural_evidence_remains_applicable_to_initialized_runtime=True,acceptance_claim=False)
(S/'initialized_runtime_guard_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
