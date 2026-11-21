import unittest,ast,json,hashlib,collections
from pathlib import Path
from group_baseline import episode_advantages
O=Path(__file__).resolve().parent;R=O.parents[1]
class TestDiagnosticContract(unittest.TestCase):
 def test_rloo_sign_and_zero_groups(self):
  values=episode_advantages([24.1,32.5,24.1,24.1]);self.assertGreater(values[0],0);self.assertLess(values[1],0);self.assertAlmostEqual(sum(values),0)
  self.assertTrue(all(abs(a)<1e-12 for a in episode_advantages([7.6]*4)))
  self.assertTrue(any(max(g)-min(g)>1e-6 for g in [[24.1,32.5,24.1,24.1],[7.6]*4,[300.]*4]))
 def test_complete_failure_retained(self):self.assertLess(episode_advantages([10.,20.,30.,300.])[-1],0)
 def test_task_split_and_contract_hashes(self):
  p=json.loads((O/'plan.json').read_text());self.assertEqual(collections.Counter(t['seed'] for t in p['update_tasks']),{1880021:4,1280022:4,1880024:4});self.assertEqual(len({t['index'] for t in p['update_tasks']}),12)
  self.assertEqual([(t['global_index'],t['seed']) for t in p['tasks']],[(2,1280022)]*4+[(3,1880024)]*4)
  self.assertEqual(hashlib.sha256((O/'plan.json').read_bytes()).hexdigest(),(O/'plan_sha256.txt').read_text())
  for f,h in p['controller_hashes'].items():self.assertEqual(hashlib.sha256((O/f).read_bytes()).hexdigest(),h)
 def test_scripts_parse_and_no_old_wall_budget(self):
  for f in ['collect.py','run_collection.py','update_once.py']:ast.parse((O/f).read_text())
  s=(O/'collect.py').read_text();self.assertIn('bind_wall_budget(bench.run,1500.)',s);self.assertNotIn('bind_wall_budget(bench.run,900.)',s)
  u=(O/'update_once.py').read_text();self.assertIn('FrozenBehaviorBatch(12)',u);self.assertIn("assert any_variance,",u);self.assertIn("assert proof['source_unchanged'] is True",u)
if __name__=='__main__':unittest.main()
