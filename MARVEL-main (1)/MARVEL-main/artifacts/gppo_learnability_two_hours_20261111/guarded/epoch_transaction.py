"""Epoch-level transaction for a model and its Adam state. No policy decisions."""
import copy,math,hashlib
import torch
class NoAcceptedEpoch(RuntimeError):pass

def state_fingerprint(state):
 digest=hashlib.sha256()
 for name,tensor in sorted(state.items()):
  v=tensor.detach().cpu().contiguous();digest.update(name.encode());digest.update(str((v.dtype,tuple(v.shape))).encode());digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
 return digest.hexdigest()

class EpochTransaction:
 def __init__(self):self.pending=None;self.history=[];self.accepted_epochs=0
 def before_epoch(self,model,optimizer,epoch):
  if self.pending is not None:raise RuntimeError('Unresolved previous epoch transaction')
  self.pending=dict(epoch=epoch,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()))
 def rollback(self,model,optimizer):
  if self.pending is None:return
  model.load_state_dict(self.pending['model']);optimizer.load_state_dict(self.pending['optimizer']);optimizer.zero_grad(set_to_none=True);self.pending=None
 def after_epoch(self,model,optimizer,epoch,metrics):
  if self.pending is None or self.pending['epoch']!=epoch:raise RuntimeError('Missing matching epoch snapshot')
  accepted=all(math.isfinite(float(metrics[k])) and float(metrics[k])<=cap for k,cap in [('sampled_kl',.01),('mean_exact_kl',.01),('max_state_exact_kl',.05)])
  accepted=accepted and all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
  row=dict(epoch=epoch,**metrics,accepted=bool(accepted),rolled_back=not accepted)
  if accepted:self.accepted_epochs+=1;self.pending=None
  else:self.rollback(model,optimizer)
  row['retained_model_fingerprint']=state_fingerprint(model.state_dict());self.history.append(row)
  return bool(accepted)
