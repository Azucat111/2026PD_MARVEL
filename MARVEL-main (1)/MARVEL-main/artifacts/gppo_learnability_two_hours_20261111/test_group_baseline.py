import unittest
from group_baseline import episode_advantages
class TestBaseline(unittest.TestCase):
 def test_sign_and_exclusion(self):
  self.assertEqual(episode_advantages([10.,20.,30.,40.]),[20./15.,(80./3-20)/15.,(70./3-30)/15.,-20./15.])
 def test_constant(self):self.assertEqual(episode_advantages([12.,12.,12.,12.]),[0.]*4)
 def test_failure_is_kept(self):self.assertLess(episode_advantages([10.,20.,30.,300.])[-1],0)
 def test_reject(self):
  for x in [[1.],[float('nan'),2.],[-1.,2.],[301.,2.]]:
   with self.assertRaises(ValueError):episode_advantages(x)
if __name__=='__main__':unittest.main()
