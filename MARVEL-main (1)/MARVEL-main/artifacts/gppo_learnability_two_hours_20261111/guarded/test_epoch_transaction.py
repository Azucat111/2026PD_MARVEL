import copy,unittest,torch,math
from epoch_transaction import EpochTransaction

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
class TestEpochTransaction(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(10);self.model=torch.nn.Linear(1,1).double();self.opt=torch.optim.Adam(self.model.parameters(),lr=.01);self.guard=EpochTransaction()
 def step(self):
  self.opt.zero_grad(set_to_none=True);self.model(torch.tensor([[2.]],dtype=torch.float64)).square().mean().backward();self.opt.step()
 def test_accept_then_reject_restores_model_and_adam(self):
  self.guard.before_epoch(self.model,self.opt,0);self.step();self.assertTrue(self.guard.after_epoch(self.model,self.opt,0,dict(sampled_kl=.003,mean_exact_kl=.002,max_state_exact_kl=.03)))
  state=copy.deepcopy(self.model.state_dict());adam=copy.deepcopy(self.opt.state_dict())
  self.guard.before_epoch(self.model,self.opt,1);self.step();self.assertFalse(equal(state,self.model.state_dict()));self.assertFalse(self.guard.after_epoch(self.model,self.opt,1,dict(sampled_kl=.0107,mean_exact_kl=.009,max_state_exact_kl=.122)))
  self.assertTrue(equal(state,self.model.state_dict()));self.assertTrue(equal(adam,self.opt.state_dict()));self.assertEqual(self.guard.accepted_epochs,1)
 def test_first_rejection_restores_initial_and_empty_adam(self):
  state=copy.deepcopy(self.model.state_dict());adam=copy.deepcopy(self.opt.state_dict());self.guard.before_epoch(self.model,self.opt,0);self.step();self.assertFalse(self.guard.after_epoch(self.model,self.opt,0,dict(sampled_kl=0.,mean_exact_kl=0.,max_state_exact_kl=.051)));self.assertTrue(equal(state,self.model.state_dict()));self.assertTrue(equal(adam,self.opt.state_dict()));self.assertEqual(self.guard.accepted_epochs,0)
 def test_each_cap_and_nonfinite_rejected(self):
  for field,value in [('sampled_kl',.01001),('mean_exact_kl',.01001),('max_state_exact_kl',.05001),('sampled_kl',float('nan')),('mean_exact_kl',float('inf'))]:
   with self.subTest(field=field,value=value):
    self.setUp();state=copy.deepcopy(self.model.state_dict());self.guard.before_epoch(self.model,self.opt,0);self.step();m=dict(sampled_kl=.001,mean_exact_kl=.001,max_state_exact_kl=.001);m[field]=value;self.assertFalse(self.guard.after_epoch(self.model,self.opt,0,m));self.assertTrue(equal(state,self.model.state_dict()))
 def test_caps_are_inclusive_and_exception_rollback(self):
  self.guard.before_epoch(self.model,self.opt,0);self.step();self.assertTrue(self.guard.after_epoch(self.model,self.opt,0,dict(sampled_kl=.01,mean_exact_kl=.01,max_state_exact_kl=.05)))
  state=copy.deepcopy(self.model.state_dict());adam=copy.deepcopy(self.opt.state_dict());self.guard.before_epoch(self.model,self.opt,1);self.step();self.guard.rollback(self.model,self.opt);self.assertTrue(equal(state,self.model.state_dict()));self.assertTrue(equal(adam,self.opt.state_dict()))
if __name__=='__main__':unittest.main()
