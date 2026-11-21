"""Thin diagnostic boundary over sklearn Ridge and the existing GPPO actor trunk."""
import numpy as np
import torch
from sklearn.linear_model import Ridge

def paired_gain(baseline_mean,branch_mean):
 if not np.isfinite([baseline_mean,branch_mean]).all() or baseline_mean<=0 or branch_mean<=0:raise ValueError('Positive finite true Find means required')
 return float(baseline_mean)-float(branch_mean)

def fit_gain_model(features,gains,alpha=10.):
 x=np.asarray(features,dtype=float);y=np.asarray(gains,dtype=float)
 if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite paired features/labels required')
 if alpha<=0 or not np.isfinite(alpha):raise ValueError('Positive finite regularization required')
 return Ridge(alpha=alpha,fit_intercept=False,solver='svd').fit(x,y)

def select_gain_action(logits,mask,actions,min_gain=.05):
 scores=np.asarray(logits,dtype=float).reshape(-1);mask=np.asarray(mask,dtype=bool).reshape(-1)
 if scores.shape!=mask.shape or not actions or not np.isfinite(min_gain) or min_gain<0:raise ValueError('Aligned scores/mask and public candidate list required')
 if any(not 0<=int(a)<len(scores) or mask[int(a)] for a in actions):raise ValueError('Invalid or masked candidate')
 if not np.isfinite(scores[actions]).all():raise ValueError('Finite candidate gains required')
 best=actions[int(np.argmax(scores[actions]))]
 return int(best if scores[best]-scores[actions[0]]>=min_gain else actions[0])

def candidate_representations(model,batch,actions):
 if model.training:raise ValueError('Frozen eval-mode GPPO representation required')
 mask=batch['action_mask'][0].flatten()
 if any(not 0<=int(a)<mask.numel() or bool(mask[int(a)]) for a in actions):raise ValueError('Legal actions required')
 captured=[]
 hook=model.actor.scorer[-1].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
 try:
  with torch.no_grad():model(**batch)
 finally:hook.remove()
 assert len(captured)==1
 return captured[0][0].reshape(-1,captured[0].shape[-1])[actions].cpu().numpy().astype(float)

def fit_bounded_gain_model(features,gains,bounded_features,upper_bounds,alpha=10.):
 """Constrained Ridge via SciPy: censored gains are inequalities, not point labels."""
 from scipy.optimize import minimize,LinearConstraint
 x=np.asarray(features,dtype=float);y=np.asarray(gains,dtype=float);c=np.asarray(bounded_features,dtype=float);u=np.asarray(upper_bounds,dtype=float)
 if x.ndim!=2 or y.shape!=(len(x),) or c.ndim!=2 or c.shape[1]!=x.shape[1] or u.shape!=(len(c),) or alpha<=0 or not all(np.isfinite(a).all() for a in [x,y,c,u]):raise ValueError('Finite aligned exact/interval labels required')
 if len(c)==0:return fit_gain_model(x,y,alpha)
 h=x.T@x+alpha*np.eye(x.shape[1]);rhs=x.T@y
 result=minimize(lambda w:.5*float(w@h@w)-float(rhs@w),np.zeros(x.shape[1]),jac=lambda w:h@w-rhs,method='SLSQP',constraints=[LinearConstraint(c,-np.inf,u)],options=dict(maxiter=500,ftol=1e-10))
 if not result.success or np.any(c@result.x>u+1e-5):raise ValueError('Constrained Ridge failed: '+str(result.message))
 fitted=Ridge(alpha=alpha,fit_intercept=False);fitted.coef_=result.x;fitted.intercept_=0.;fitted.n_features_in_=x.shape[1];fitted.n_iter_=int(result.nit)
 return fitted
